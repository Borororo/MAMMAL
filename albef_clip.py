import argparse
import os
try:
    import ruamel_yaml as yaml
except:
    import ruamel.yaml as yaml
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

from models.model_vqa_clip import ALBEF
from models.vit import interpolate_pos_embed, resize_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset.utils import save_result
from dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn,clevr_collate_fn

from scheduler import create_scheduler
from optim import create_optimizer, create_two_optimizer


def train_clevr(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config,do_amp=False,do_two_optim=False,do_three_optim=False, accum_steps=1):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if do_two_optim:
        metric_logger.add_meter('lr1', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
        metric_logger.add_meter('lr2', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    elif do_three_optim:
        metric_logger.add_meter('lr1', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
        metric_logger.add_meter('lr2', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
        metric_logger.add_meter('lr3', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    else:
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['use_gt_layout']:
        metric_logger.add_meter('layout_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size
#  torch.stack(image_list,dim=0), question_list,question_ids,answer_list, n,torch.stack(gt_label_list,dim=0),torch.stack(answer_label_list,dim=0)
    for i, (image, question, qids,answer, weights, n, layout_label, soft_answer_label) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, weights = image.to(device, non_blocking=True), weights.to(device, non_blocking=True)
        question_input = tokenizer(question, padding='longest', truncation=True, max_length=args.max_input_length if config["add_ocr"] else 25, return_tensors="pt").to(
            device)
        # if i == 0:
        #     print ("question: ", question)
        answer_input = tokenizer(answer, padding='longest', return_tensors="pt").to(device)
        layout_label = layout_label.to(device)
        soft_answer_label = soft_answer_label.to(device)
        if i == 0:
                print ("question: ", question)
                print("answer",answer)
                print("weights",weights)
                print("n",n)
        if epoch > 0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        loss = model(image, question_input, answer_input, train=True, alpha=alpha, k=n, weights=weights)
 

        # if config['use_gt_layout']:
        #     layout_loss = compute_gt_layout_loss(config,outputs['module_logits'],layout_label)
        #     loss += layout_loss
        
        optimizer.zero_grad()
        # loss.backward()
        
        if args.do_amp:
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                # logger.info('scaled loss: {}'.format(str(scaled_loss)))
                scaled_loss.backward()
        else:
            loss.backward()
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name)
        optimizer.step()    


        metric_logger.update(loss=loss.item())

        if do_two_optim:
            metric_logger.update(lr1=optimizer.param_groups[0]["lr"])
            metric_logger.update(lr2=optimizer.param_groups[2]["lr"])
        elif do_three_optim:
            metric_logger.update(lr1=optimizer.param_groups[0]["lr"])
            metric_logger.update(lr2=optimizer.param_groups[2]["lr"])
            metric_logger.update(lr3=optimizer.param_groups[4]["lr"])
        else:
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if config['use_gt_layout']:
            metric_logger.update(layout_loss=layout_loss.item())
        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)


    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, do_amp=False,
          do_two_optim=False, do_accum=False, accum_steps=1):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if do_two_optim:
        metric_logger.add_meter('lr1', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
        metric_logger.add_meter('lr2', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    else:
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    for i, (image, question, answer, weights, n) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, weights = image.to(device, non_blocking=True), weights.to(device, non_blocking=True)
        question_input = tokenizer(question, padding='longest', truncation=True, max_length=args.max_input_length if config["add_ocr"] else 25, return_tensors="pt").to(
            device)

        answer_input = tokenizer(answer, padding='longest', return_tensors="pt").to(device)
        if i == 0:
            print ("question: ", question)
            print("answer",answer)
            print("weights",weights)
            print("n",n)

        if epoch > 0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        loss = model(image, question_input, answer_input, train=True, alpha=alpha, k=n, weights=weights)
        #if accum_steps > 1:
        #    loss = loss / accum_steps
        optimizer.zero_grad()
        # loss.backward()
        if do_amp:
           from apex import amp
           with amp.scale_loss(loss, optimizer) as scaled_loss:
               # logger.info('scaled loss: {}'.format(str(scaled_loss)))
               scaled_loss.backward()
        else:
           loss.backward()

        for name, param in model.named_parameters():
            if param.grad is None:
                print(name)
        #if (i + 1) % accum_steps == 0:
        #    optimizer.step()
        #    optimizer.zero_grad()
        # model.backward(loss)
        # model.step()
        optimizer.step()    


        metric_logger.update(loss=loss.item())
        # metric_logger.update(loss=loss.item())

        if do_two_optim:
            metric_logger.update(lr1=optimizer.param_groups[0]["lr"])
            metric_logger.update(lr2=optimizer.param_groups[2]["lr"])
        else:
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)
        del image,weights, question_input,answer_input, loss
            # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluation_clevr(model, data_loader, tokenizer, device, config) :
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
                print(ans)
                valid_answer = ans.split()[0]
                if ans.split()[0] == "cyananananananananan" or ans.split()[0].startswith("cyan"):
                    valid_answer = "cyan"
                if ans.split()[0] == "purpleananananan" or ans.split()[0].startswith("purple"):
                    valid_answer = "purple"
                # print({"question_id":ques_id, "pred":valid_answer,"ref":ref_a})
                result.append({"question_id":ques_id, "pred":valid_answer,"ref":ref_a})   

            else:
                _, pred = topk_prob.max(dim=0)
                result.append({"question_id":ques_id, "pred":data_loader.dataset.valid_answer_list[topk_id[pred]],"ref":ref_a})  
                print({"question_id":ques_id, "pred":data_loader.dataset.valid_answer_list[topk_id[pred]],"ref":ref_a}) 


    return result

@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50

    result = []

    answer_list = [answer + config['eos'] for answer in data_loader.dataset.answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)

    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)

        topk_ids, topk_probs = model(image, question_input, answer_input, train=False, k=config['k_test'])

        for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
            ques_id = int(ques_id.item())          
            if config['open_generation']:
 
                ans = tokenizer.decode(topk_id).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
                print(ans)
                result.append({"question_id":ques_id, "answer":ans})   
            else:
                _, pred = topk_prob.max(dim=0)
                result.append({"question_id":ques_id, "answer":data_loader.dataset.answer_list[topk_id[pred]]})   

    return result

@torch.no_grad()
def evaluate(model, data_loader, dataset, tokenizer, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50
    result = []
    
    answer_list = [answer+config['eos'] for answer in data_loader.dataset.answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)    
    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device,non_blocking=True)             
        question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)        

        topk_ids, topk_probs = model(image, question_input, answer_input, train=False, k=config['k_test'])      
        
        for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
            ques_id = int(ques_id.item())          
            if config['open_generation']:
                ans = tokenizer.decode(topk_id).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
                result.append({"question_id":ques_id, "answer":ans})   
            else:
                _, pred = topk_prob.max(dim=0)
                result.append({"question_id":ques_id, "answer":data_loader.dataset.answer_list[topk_id[pred]]})   
        accuracy = cal_metric(result, dataset)
        # accuracy = (targets == pred_class).sum() / targets.size(0)
        #
        metric_logger.meters['acc'].update(accuracy, n=image.size(0))

    # gather the stats from all processes
    torch.cuda.empty_cache()
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
def cal_metric(vqa_result, val_file):
    
    with open(val_file[0], "r") as f:
        data_list = json.load(f)
    id2datum = {}
    for each in data_list:
        id2datum[each["question_id"]] = each["label"]
    score = 0.
    for each in vqa_result:
        quesid = each["question_id"]
        ans = each["answer"]
        label = id2datum[quesid]
        if ans in label:
            score += label[ans]
    return score / len(vqa_result)

def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)
    # print('master addr: ', os.environ['MASTER_ADDR'])
    # print('master port: ', os.environ['MASTER_PORT'])
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Dataset ####
    if args.task =="clevr":
        print("Creating clevr datasets")
        datasets = create_dataset('clevr', config)

        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)
        else:
            samplers = [None, None]

        train_loader, test_loader = create_loader(datasets,samplers,
                                                batch_size=[config['batch_size_train'],config['batch_size_test'], config['batch_size_test']],
                                                num_workers=[4,4],is_trains=[True, False],
                                                collate_fns=[clevr_collate_fn,None,None])
    else:
        print("Creating vqa datasets")
        datasets = create_dataset('vqa', config)

        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)
        else:
            samplers = [None, None, None]

        train_loader, val_loader, test_loader = create_loader(datasets,samplers,
                                                batch_size=[config['batch_size_train'],config['batch_size_test'], config['batch_size_test']],
                                                num_workers=[8,8,8],is_trains=[True, False, False],
                                                collate_fns=[vqa_collate_fn,None,None])

    # for i,(image, question, answer, weights, n) in enumerate(train_loader):
    #     for ans in answer:
    #         if ans not in train_loader.dataset.answer_list:
    #             print(ans)
        # print([f in train_loader.dataset.answer_list for f in answer])

    #### Model ####
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, text_decoder=args.text_decoder, tokenizer=tokenizer)
    model = model.to(device)  
    print(model.state_dict().keys())
    if not args.do_two_optim:
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
    else:
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_two_optimizer(arg_opt, model)

    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    if args.do_amp:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        try:
            state_dict = checkpoint['model']
        except:
            state_dict = checkpoint['module']

        # reshape positional embedding to accomodate for image resolution change

        if not args.evaluate:
            if "clip_name" in config and "ViT-L-14" not in config["clip_name"]:
                num_patches = int(config["image_res"] * config["image_res"]/(16*16))
            else:
                num_patches = int(config["image_res"] * config["image_res"]/(14*14))
            pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768).float())

            pos_embed = resize_pos_embed(state_dict['visual_encoder.visual.positional_embedding'].unsqueeze(0),
                                                   pos_embed.unsqueeze(0))
            state_dict['visual_encoder.visual.positional_embedding'] = pos_embed
            if config['distill']:
                if "clip_name" in config and "ViT-L-14" not in config["clip_name"]:
                    num_patches = int(config["image_res"] * config["image_res"]/(16*16))
                else:
                    num_patches = int(config["image_res"] * config["image_res"]/(14*14))
                pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768).float())

                pos_embed = resize_pos_embed(state_dict['visual_encoder_m.visual.positional_embedding'].unsqueeze(0),
                                             pos_embed.unsqueeze(0))
                state_dict['visual_encoder_m.visual.positional_embedding'] = pos_embed

            for key in list(state_dict.keys()):
                if 'bert' in key and 'decode' not in key:
                    encoder_key = key.replace('bert.', '')
                    state_dict[encoder_key] = state_dict[key]
                    # intialize text decoder as multimodal encoder (last 6 layers of model.text_encoder)
                if 'visual' in key:
                    encoder_key = key.replace('.visual.','.')
                    state_dict[encoder_key] = state_dict[key]
                if 'text_encoder' in key:
                    if 'layer' in key:
                        encoder_keys = key.split('.')
                        layer_num = int(encoder_keys[4])
                        if layer_num < 6:
                            del state_dict[key]
                            continue
                        else:
                            decoder_layer_num = (layer_num - 6)
                            encoder_keys[4] = str(decoder_layer_num)
                            encoder_key = '.'.join(encoder_keys)
                    else:
                        encoder_key = key
                    decoder_key = encoder_key.replace('text_encoder', 'text_decoder')
                    if not args.no_init_decocde and not args.has_decode:
                        state_dict[decoder_key] = state_dict[key]
                    else:
                        #if 'embeddings' in key:
                        #    state_dict[decoder_key] = state_dict[key]
                        pass
                        

                    del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        # print(msg)
    # exit()
    model_without_ddp = model
    if args.distributed:
        if args.do_amp:
            import apex
            model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    

    # if args.deepspeed:
    #     model, optimizer, _, lr_schedule = deepspeed.initialize(
    #         model=model,
    #         optimizer=optimizer,
    #         args=args,
    #         lr_scheduler=lr_scheduler,
    #         dist_init_required=True
    #     )

    # device = torch.device(args.device + ':' + os.environ['LOCAL_RANK'])
    # print('local device:', device)




    print("Start training")
    start_time = time.time()
    print(f"number epoch {max_epoch}")
    for epoch in range(start_epoch, max_epoch):
        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)

        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            if args.task =="clevr":
                train_stats = train_clevr(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler,
                                config, do_amp=args.do_amp, do_two_optim=args.do_two_optim, accum_steps=args.accum_steps)
            else:
                train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler,
                                    config, do_amp=args.do_amp, do_two_optim=args.do_two_optim, accum_steps=args.accum_steps)
        # if int(config["image_res"]) != 504:
        #     val_stats = evaluate(model, val_loader, config["label_file"], tokenizer, device, config)
        #     model.save_checkpoint(os.path.join(args.output_dir), tag='{}.pt'.format(model.global_steps))
        # if epoch >= 5:
        #     vqa_result = evaluation(model, test_loader, tokenizer, device, config)
        #     result_file = save_result(vqa_result, args.result_dir, 'vqa_result_epoch%d' % epoch)

            
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
    if args.task == 'clevr':
        vqa_result = evaluation_clevr(model,test_loader,tokenizer,device,config)
        result_file = save_result(vqa_result, args.result_dir, 'vqa_result_epoch%d'%epoch)
        predictions = json.load(open(result_file,'r'))
        total = 0
        acc = 0
        for ins in predictions:
            if  ins['pred'] == ins['ref']:
                acc+=1
            total +=1
        print(f'final accuracy {acc/total}')
        
    else:
        vqa_result = evaluation(model,test_loader,tokenizer,device,config)
        result_file = save_result(vqa_result, args.result_dir, 'vqa_result_epoch%d'%epoch)
    # vqa_result = evaluation(model, test_loader, tokenizer, device, config)
    # result_file = save_result(vqa_result, args.result_dir, 'vqa_result_epoch%d' % epoch)

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
    config["open_generation"] = args.open_generation
    config["min_length"] = args.min_length
    config["max_length"] = args.max_length
    config["beam_size"] = args.beam_size
    config['merge_attention'] = args.merge_attention
    config['concat_last_layer'] = args.concat_last_layer
    config['schedular']['epochs'] = args.epoch
    config['add_ocr'] = args.add_ocr
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
    
    try:     
        yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    except:
        pass
    main(args, config)
