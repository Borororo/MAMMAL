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

# # from models.model_vqa_clip import ALBEF
# from models.model_vqa_clip_nmn_generation import ALBEF as ALBEF_nmn
# from models.model_vqa_clip_transnmn_generation import ALBEF as ALBEF_transnmn
from models.model_vqa_clip_nmn_generation import ALBEF as ALBEF_nmn
from models.model_vqa_clip_transnmn_generation import ALBEF
from models.vit import interpolate_pos_embed, resize_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset.utils import save_result
from dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn,vqa_nmn_collate_fn

from scheduler import create_scheduler
from optim import create_optimizer, create_two_optimizer,create_three_optimizer

import matplotlib.pyplot as plt
import skimage.io
import skimage.transform

@torch.no_grad()
def visualize_clevr(model, data_loader, tokenizer,device,config,args):
     # test
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50
    module_names= config['module_names']
    answer_list = [answer+config['eos'] for answer in data_loader.dataset.valid_answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)
    clean_answer_list =  [answer for answer in data_loader.dataset.valid_answer_list]

    for n, (image, question, question_id,image_path,answer) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):   
        if n>5:
            break
        image = image.to(device,non_blocking=True)             
        question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)      
        topk_ids, topk_probs,outputs = model(image, question_input, answer_input, train=False, k=config['k_test'])
        predicts = []
        for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):          
            ans = tokenizer.decode(topk_id).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
            predicts.append(ans)   
  
      
        # visualize_stepwise(outputs,image_path,args.vis_dir,question,question_id,question_input,clean_answer_list,tokenizer,module_names,layout_label)    

        visualize_stepwise(outputs,image_path,args.vis_dir,question,question_id,question_input,clean_answer_list,tokenizer,module_names,predicts,config)


@torch.no_grad()
def visualize_vqa(model, data_loader, tokenizer,device,config,args):
     # test
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50
    module_names= config['module_names']
    answer_list = [answer+config['eos'] for answer in data_loader.dataset.valid_answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)
    clean_answer_list =  [answer for answer in data_loader.dataset.valid_answer_list]
    for n, (image, question, question_id,image_path) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if n >5:
            break
        image = image.to(device,non_blocking=True)             
        question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)    

        topk_ids, topk_probs,outputs = model(image, question_input, answer_input, train=False, k=config['k_test'])
        predicts = []
        for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):          
            ans = tokenizer.decode(topk_id).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
            predicts.append(ans)   

        # loss,outputs = model(image, question_input, answer_input, train=False, k=config['k_test'])
        # print(question_input)
        visualize_stepwise(outputs,image_path,args.vis_dir,question,question_id,question_input,clean_answer_list,tokenizer,module_names,predicts,config)


def visualize_stepwise(outputs,image_path,vis_root,question,question_id,question_input,answer_list,tokenizer,module_names,ans,config):
    T = outputs['qattns'].size(1)
    B = outputs['qattns'].size(0)
    H = config['h_feature']
    W = config['w_feature']
    # D = outputs['iattns'].size(-1)
    question_attns = outputs['qattns'].detach().cpu().numpy()
    # print(f"image attn shape {outputs['iattns'].shape}")
    # print(f"image attn {torch.sum(outputs['iattns'],dim=-1)}")
    # image_attns = outputs['iattns'].detach().cpu().numpy()
    if len(outputs['iattns'].shape)==4:
        image_attns = outputs['iattns'].detach().cpu().numpy()

    else:
        image_attns = outputs['iattns'].view(B,T,H,W).detach().cpu().numpy()
    
    module_probs = outputs["module_logits"].softmax(dim=-1).detach().cpu().numpy()
    # prediction_probs = outputs["answer_logits"].softmax(dim=-1).detach().cpu().numpy()
    attn_stacks = outputs['att_stack'].detach().cpu().numpy() 
    pointer_stacks = outputs['stack_ptr'].detach().cpu().numpy()
    question_input_ids = question_input['input_ids'].detach().cpu()

  

    def attention_interpolation(im, att):
        softmax = att_softmax(att)
        # print(im.shape)
        # print(softmax.shape)
        # print(np.sum(softmax,-1))
        att_reshaped = skimage.transform.resize(att, im.shape[:2], order=3)
        # normalize the attention
        # make sure the 255 alpha channel is at least 3x uniform attention
        att_reshaped /= np.maximum(np.max(att_reshaped), 3. / att.size)
        att_reshaped = att_reshaped[..., np.newaxis]

        # make the attention area brighter than the rest of the area
        # vis_im = att_reshaped * im 
        vis_im = att_reshaped * im + (1-att_reshaped) * im * .35
        vis_im = vis_im.astype(im.dtype)
        return vis_im

    def att_softmax(att):
        exps = np.exp(att - np.max(att))
        softmax = exps / np.sum(exps)
        return softmax
  
    for n, (question_attn,image_attn,answer, module_prob,attn_stack,ptr_stack,question_input_id) in enumerate(zip(question_attns, image_attns, ans, module_probs, attn_stacks,pointer_stacks,question_input_ids)):
        current_question = question[n]
        # question_input = tokenizer(question, padding='longest', return_tensors="pt")
        question_tok = tokenizer.convert_ids_to_tokens(question_input_id) 
        img_path = image_path[n]
        q_id =question_id[n]
        # gt_layout = layout_label[n]
        img = skimage.io.imread(img_path)
        h = plt.figure(figsize=(20, 20))
        plt.subplot(4,4,1)
        plt.imshow(img)
        plt.title(current_question+f' \n prediction: {answer}')
        # module weights
        plt.subplot(4,4, 2)
        plt.imshow(np.array(module_prob).T, cmap='Reds')
        plt.colorbar()
        plt.xticks(range(T), range(T))
        plt.yticks(range(len(module_names)), module_names, size='small')
        plt.title('module weights at controller timestep')

        # moduel weights

        # plt.subplot(4,4, 3)
        # module_prob_gt = np.zeros([len(gt_layout),len(module_names)])
        # for ind,label in enumerate(gt_layout):
        #     module_prob_gt[ind,label]=1.0
        # plt.imshow(np.array(module_prob_gt).T, cmap='Reds')
        # plt.colorbar()
        # plt.xticks(range(T), range(T))
        # plt.yticks(range(len(module_names)), module_names, size='small')
        # plt.title('Ground Truth Layout')

        # textual attention
        plt.subplot(4,4, 3)
        # print(question_tok)
        # print(f"question-attn shape:{question_attn.shape}")
        plt.imshow(question_attn, cmap='Reds')
        plt.colorbar()
        plt.xticks(range(len(question_tok)), question_tok,rotation=90)
        plt.yticks(range(T), range(T))
        plt.ylabel('controller timestep')
        plt.title('textual attention at controller timestep')


        # print(prt_stack)
        plt.subplot(4,4, 4)
        plt.imshow(ptr_stack.T, cmap='Reds')
        # print(ptr_stack.shape)
        plt.colorbar()
        plt.xticks(range(T), range(T))
        plt.yticks(range(ptr_stack.shape[1]), range(ptr_stack.shape[1]))
        plt.ylabel('stack depth')
        plt.xlabel('stack pointer at controller timestep')

        for t in range(T):
            plt.subplot(4,4, t+5)
            img_with_att = attention_interpolation(img, image_attn[t])
            plt.imshow(img_with_att)
            plt.xlabel('controller timestep t = %d' % t)

        
        out_path = str(q_id.numpy())+ '_'+img_path.split('/')[-1]
        save_path =os.path.join(vis_root,out_path)

        plt.savefig(save_path)
        print('visualization saved to ' + save_path)
        plt.close(h)


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    if args.task =="vqa":
        print("Creating vqa datasets")
        datasets = create_dataset('vqa-vis', config)   

        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()            
            samplers = create_sampler(datasets, [False, False], num_tasks, global_rank)         
        else:
            samplers = [None, None]


        val_loader, test_loader = create_loader(datasets,samplers,
                                                batch_size=[config['batch_size_test'],config['batch_size_test']],
                                                num_workers=[4,4],is_trains=[False, False], 
                                                collate_fns=[None,None]) 
    
    if args.task =="clevr" or args.task =="clevr_nmn":
        print("Creating clevr datasets")
        datasets = create_dataset('clevr', config)   
        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()            
            samplers = create_sampler(datasets, [False, False], num_tasks, global_rank)  
        else:
            samplers = [None, None]
 
        train_loader, test_loader = create_loader(datasets,samplers,
                                            batch_size=[config['batch_size_train'],config['batch_size_test']],
                                            num_workers=[8,4],is_trains=[False, False], 
                                            collate_fns=[None,None]) 

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model #### 
    print("Creating model")
    if args.task =="clevr_nmn":
        model = ALBEF_nmn(config=config, text_encoder=args.text_encoder, text_decoder=args.text_decoder, tokenizer=tokenizer,task='9')
    else:
        model = ALBEF(config=config, text_encoder=args.text_encoder, text_decoder=args.text_decoder, tokenizer=tokenizer)
    model = model.to(device)         
  
    # print(model.state_dict().keys())
    # exit()
    # for name, param in model.named_parameters():
    #     # if param.grad is None:
    #     print(name)
    if  args.do_two_optim:
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_two_optimizer(arg_opt, model)
    elif args.do_three_optim:
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_three_optimizer(arg_opt, model)
    else:
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)

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

        print("preparing state dict done!!")
        # for k in state_dict.keys():
        #     if 'visual' in k:
        #         print(k)
        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    model_without_ddp = model
    if args.distributed:
        if args.do_amp:
            import apex
            model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    


    print("Start visualizing")
    start_time = time.time()

    if args.task =="vqa":
        if args.split.startswith('val'):
            visualize_vqa(model,val_loader,tokenizer,device,config,args)
        else:
            visualize_vqa(model,test_loader,tokenizer,device,config,args)

    if args.task.startswith("clevr"):
            visualize_clevr(model,test_loader,tokenizer,device,config,args)
  
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Visualizing time {}'.format(total_time_str))


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
    parser.add_argument('--vis_dir', default='/nas-alinlp/lcl193798/data_renm/vqa_result') 
    parser.add_argument('--split', default='val') 
    parser.add_argument('--layout', action='store_true')  
    parser.add_argument('--replace_fusion', action='store_true')
    parser.add_argument('--use_image_cls', action='store_true')
    parser.add_argument('--use_cat_cls', action='store_true') 
    parser.add_argument('--use_mean_cls', action='store_true')     
    parser.add_argument('--mask_pad', action='store_true')    
    parser.add_argument('--batch_size_test', default=16, type=int)
    parser.add_argument('--batch_size_train', default=16, type=int)
    parser.add_argument('--task', default='vqa')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')
    args.vis_dir = os.path.join(args.vis_dir, args.split)
    Path(args.vis_dir).mkdir(parents=True, exist_ok=True)
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
    
    try:     
        yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    except:
        pass
    main(args, config)