import argparse
import os
try:
    import ruamel_yaml as yaml
except:
    import ruamel.yaml as yaml
# import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist


from models.model_vqa import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer


import utils
from dataset.utils import save_result
from dataset import create_dataset, create_sampler, create_loader,clevr_collate_fn
from scheduler import create_scheduler
from optim import create_optimizer

def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config,args):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50    
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    for i, (image, question, qids,answer, weights, n, layout_label, soft_answer_label) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, weights = image.to(device,non_blocking=True), weights.to(device,non_blocking=True)      
        question_input = tokenizer(question, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device) 
        answer_input = tokenizer(answer, padding='longest', return_tensors="pt").to(device) 
       
        if epoch > 0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        loss = model(image, question_input, answer_input, train=True, alpha=alpha, k=n, weights=weights)   
 
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name)

        
        if config['accum_steps'] > 1:
           loss = loss / config['accum_steps']

        if args.do_amp:
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                # logger.info('scaled loss: {}'.format(str(scaled_loss)))
                scaled_loss.backward()
        else:
            loss.backward()

        if (i + 1) % config['accum_steps'] == 0:
            if config['clip_gradient']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()
 
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size) 
            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} 


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config) :
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print_freq = 50
    result = []
    
    answer_list = [answer+config['eos'] for answer in data_loader.dataset.valid_answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)    
    for n, (image, question, question_id,image_path,answer) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device,non_blocking=True)             
        question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)        

        topk_ids, topk_probs= model(image, question_input, answer_input, train=False, k=config['k_test'])      
        # print(len(question_id))
        # print(len(topk_ids))
        # print(answer)
        # print(len(answer[0]))
        # print(topk_ids,topk_probs)
        for ques_id, topk_id, topk_prob,ref_a in zip(question_id, topk_ids, topk_probs,answer[0]):
            ques_id = int(ques_id.item())          
            if config['open_generation']:
                ans = tokenizer.decode(topk_id).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
                # valid_ans = None
                # for a in ans.split():
                #     if a in data_loader.dataset.valid_answer_list:
                #         valid_ans = a
                #         break
                # print({"question_id":ques_id, "pred":ans.split()[0],"ref":ref_a,"valid":valid_ans})
                valid_answer = ans.split()[0]
                if ans.split()[0] == "cyananananananananan" or ans.split()[0].startswith("cyan"):
                    valid_answer = "cyan"
                # print({"question_id":ques_id, "pred":valid_answer,"ref":ref_a})
                result.append({"question_id":ques_id, "question":question, "pred":valid_answer,"ref":ref_a})   

            else:
                # print(topk_id,topk_prob)
                _, pred = topk_prob.max(dim=0)
                result.append({"question_id":ques_id, "pred":data_loader.dataset.valid_answer_list[topk_id[pred]],"ref":ref_a})  
                print({"question_id":ques_id, "pred":data_loader.dataset.valid_answer_list[topk_id[pred]],"ref":ref_a}) 


    return result



def main(args, config):
    utils.init_distributed_mode(args)    
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    
    #### Dataset #### 

    print("Creating clevr datasets")
    datasets = create_dataset('clevr', config)   

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)         
    else:
        samplers = [None, None]
    
    train_loader, test_loader = create_loader(datasets,samplers,
                                            batch_size=[config['batch_size_train'],config['batch_size_test']],
                                            num_workers=[4,4],is_trains=[True, False], 
                                            collate_fns=[clevr_collate_fn,None]) 

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)



    #### Model ####
    model = ALBEF(config=config, text_encoder=args.text_encoder, text_decoder=args.text_decoder, tokenizer=tokenizer)
    model = model.to(device)   

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)          
        
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        try:
            state_dict = checkpoint['model']
        except:
            state_dict = checkpoint
        
        try:
            # reshape positional embedding to accomodate for image resolution change
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped   
        except:
            pass
        if not args.evaluate:
            if config['distill']:
                m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
                state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 
                
            for key in list(state_dict.keys()):
                if 'bert' in key:
                    encoder_key = key.replace('bert.','')         
                    state_dict[encoder_key] = state_dict[key]         
                   # intialize text decoder as multimodal encoder (last 6 layers of model.text_encoder) 
                if "ag" in args.task: 
                    if 'text_encoder' in key:                
                        if 'layer' in key:
                            encoder_keys = key.split('.')
                            layer_num = int(encoder_keys[4])
                            if layer_num<6:
                                del state_dict[key]  
                                continue
                            else:
                                decoder_layer_num = (layer_num-6)
                                encoder_keys[4] = str(decoder_layer_num)
                                encoder_key = '.'.join(encoder_keys)     
                        else:
                            encoder_key = key
                        decoder_key = encoder_key.replace('text_encoder','text_decoder')  
                        state_dict[decoder_key] = state_dict[key]     

                        del state_dict[key]                       
        msg = model.load_state_dict(state_dict,strict=False)  
        print('load checkpoint from %s'%args.checkpoint)
        # print(msg)  

        
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    
    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps)  
        
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config,args)
    
        if args.evaluate:
            break
            
        if utils.is_main_process():                                                     
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  

        dist.barrier()   

    vqa_result = evaluation(model,test_loader,tokenizer,device,config)
    result_file = save_result(vqa_result, args.result_dir, 'vqa_result_epoch%d'%epoch)
    predictions = json.load(open(result_file,'r'))
    total = 0
    acc = 0
    for ins in predictions:
        if  ins['pred'] == ins['ref']:
            acc+=1
        total +=1
    print(f'final accuracy {acc/total}')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VQA.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output_dir', default='output/vqa')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--decoder_method', default='planA')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--min_length', default=1, type=int)
    parser.add_argument('--epoch', default=8, type=int)
    parser.add_argument('--max_length', default=10, type=int)
    parser.add_argument('--max_input_length', default=50, type=int)
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--do_two_optim', action='store_true')
    parser.add_argument('--do_three_optim', action='store_true')
    parser.add_argument('--do_amp', action='store_true')
    parser.add_argument('--no_init_decocde', action='store_true')
    parser.add_argument('--open_generation', action='store_true')
    parser.add_argument('--merge_attention', action='store_true')
    parser.add_argument('--do_accum', action='store_true')
    parser.add_argument('--concat_last_layer', action='store_true')
    parser.add_argument('--add_ocr', action='store_true')
    parser.add_argument('--add_object', action='store_true')
    parser.add_argument('--has_decode', action='store_true')
    parser.add_argument('--accum_steps', default=1, type=int)
    parser.add_argument('--decode_layers', default=6, type=int)
    parser.add_argument('--task', default='vqa')
    parser.add_argument('--type', default='transnmn')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--clip_gradient', action='store_true')
    
    parser.add_argument('--layout', action='store_true')  
    parser.add_argument('--replace_fusion', action='store_true')
    parser.add_argument('--use_image_cls', action='store_true')
    parser.add_argument('--use_cat_cls', action='store_true') 
    parser.add_argument('--use_mean_cls', action='store_true')     
    parser.add_argument('--mask_pad', action='store_true')    
    parser.add_argument('--batch_size_test', default=16, type=int)
    parser.add_argument('--batch_size_train', default=16, type=int)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    config['accum_steps'] = args.accum_steps
    config["open_generation"] = args.open_generation
    config["min_length"] = args.min_length
    config["max_length"] = args.max_length
    config["beam_size"] = args.beam_size
    config['merge_attention'] = args.merge_attention
    config['concat_last_layer'] = args.concat_last_layer
    config['schedular']['epochs'] = args.epoch
    config['add_ocr'] = args.add_ocr
    config['decoder_method'] = args.decoder_method
    config['add_object'] = args.add_object
    config['has_decode'] = args.has_decode
    config['decode_layers'] = args.decode_layers
    config['use_gt_layout'] = True if args.layout else False
    config['batch_size_test'] = max(config['batch_size_test'],args.batch_size_test)
    config['batch_size_train'] = args.batch_size_train if args.batch_size_train else config['batch_size_train']
    if args.replace_fusion:
        config['replace_fusion'] = True
    config['use_image_cls'] = True if args.use_image_cls else False
    config['use_cat_cls'] = True if args.use_cat_cls else False
    config['use_mean_cls'] = True if args.use_mean_cls else False
    config['mask_pad'] = True if args.mask_pad else False
    config['clip_gradient'] = True if args.clip_gradient else False
    
    try:     
        yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    except:
        pass
    main(args, config)
