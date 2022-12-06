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
import cv2
from torchvision import transforms
from PIL import ImageFont, ImageDraw,Image

def load_image(image_path,args):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    image_transform = transforms.Compose([
            transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
            ])
    raw_image = Image.open(image_path)
    if args.save_raw:
        out_path = image_path.split('/')[-1]
        save_path =os.path.join(args.vis_dir,out_path)
        raw_image.save(save_path)

    raw_image = raw_image.convert('RGB')   
    image = image_transform(raw_image)
    return image.unsqueeze(0) 

@torch.no_grad()
def visualize_demo(model,question,image_path,tokenizer,device,args):
    model.eval()

    header = 'Generate VQA test result:'
    print_freq = 50
    module_names=['Idle',
                'Find',   
                'Transform',
                'Filter',
                'And',
                "Or",
                "Scene",
                "Answer",
                "Compare"
                ]
    # module_names= config['module_names']
    question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)

    image =load_image(image_path,args)
    image = image.to(device, non_blocking=True)  
    topk_ids, topk_probs,outputs = model(image, question_input, None, train=False, k=config['k_test'])

    predicts = []
    topk_id =topk_ids[0]
    ans = tokenizer.decode(topk_id).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip() 

    print(ans)
    visualize_stepwise_v2(outputs,image_path,question,question_input,ans,tokenizer,module_names,args)

                    

def visualize_stepwise(outputs,image_path,question,question_input,ans,tokenizer,module_names,args):
    T = outputs['qattns'].size(1)
    B = outputs['qattns'].size(0)
    H = config['h_feature']
    W = config['w_feature']
    # img_h, img_w = raw_image.size[0], raw_image.size[1]
    if len(outputs['iattns'].shape)==4:
        image_attns = outputs['iattns'].detach().cpu().numpy()
    else:
        image_attns = outputs['iattns'].view(B,T,H,W).detach().cpu().numpy()
    module_prob = outputs["module_logits"].softmax(dim=-1).detach().cpu().numpy()[0]
    # print(outputs['qattns'].shape)
    question_attn = outputs['qattns'].detach().cpu().numpy()[0]
    question_input_id = question_input['input_ids'].detach().cpu()[0]
    question_tok = tokenizer.convert_ids_to_tokens(question_input_id) 
    def att_softmax(att):
        exps = np.exp(att - np.max(att))
        softmax = exps / np.sum(exps)
        return softmax
    
    img = skimage.io.imread(image_path)

    h = plt.figure(figsize=(28, 28))
    plt.suptitle(question+f' \n prediction: {ans}')   
    for t in range (T):
        step = t+1

        ######## Image attention
        plt.subplot(3,5,t+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title('image attention controller timestep t = %d' % step)
        

        mask = att_softmax(image_attns[0][t])
        mask =skimage.transform.resize(mask,img.shape[:2],order=3)
        normed_mask = mask / mask.max()
        normed_mask = (normed_mask * 255).astype('uint8')
        plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap="jet")

        # current_action = module_names[np.argmax(np.array(module_prob[t]))]
        # current_text = question_tok['']
        # plt.xlabel('stack pointer at controller timestep')
        ###########
        plt.subplot(3,5,t+1+12)
        plt.imshow([np.array(module_prob[t])], cmap='Reds')
        # plt.colorbar()
        plt.xticks(range(len(module_names)), module_names, size='small',rotation=90)
        plt.title('module weights at controller timestep %d' % step )

        # ############
        plt.subplot(3,5,t+1+24)
        plt.imshow([question_attn[t]], cmap='Reds')
        # plt.colorbar()
        plt.xticks(range(len(question_tok)), question_tok,rotation=90)
        plt.title('textual attention at controller timestep %d' % step)

    out_path = str(question)+ '_'+image_path.split('/')[-1]
    save_path =os.path.join(args.vis_dir,out_path)
    plt.savefig(save_path)
    print('visualization saved to ' + save_path)
    plt.close(h)


    
    # mask = att_softmax(image_attns[0])
    # mask = cv2.resize(mask,raw_image.shape[:2])

    
    # # out_path = str(q_id.numpy())+ '_'+img_path.split('/')[-1]
    # # save_path =os.path.join(vis_root,out_path)
    # save_path ="//nas-alinlp/lcl193798/data_renm/CLIP_result/test.jpg"
    # plt.savefig(save_path)
    # print('visualization saved to ' + save_path)
    # plt.close(h)

    # for t in range(T):
    #     plt.subplot(1,13, t+2)
    #     mask = cv2.(image_attn,)
    #     img_with_att = attention_interpolation(img, image_attn[t])
    #     plt.imshow(img_with_att)
    #     plt.xlabel('controller timestep t = %d' % t)
    


def visualize_stepwise_v2(outputs,image_path,question,question_input,ans,tokenizer,module_names,args):
    T = outputs['qattns'].size(1)
    B = outputs['qattns'].size(0)
    H = config['h_feature']
    W = config['w_feature']
    # img_h, img_w = raw_image.size[0], raw_image.size[1]
    if len(outputs['iattns'].shape)==4:
        image_attns = outputs['iattns'].detach().cpu().numpy()
    else:
        image_attns = outputs['iattns'].view(B,T,H,W).detach().cpu().numpy()
    module_prob = outputs["module_logits"].softmax(dim=-1).detach().cpu().numpy()[0]
    # print(outputs['qattns'].shape)
    question_attn = outputs['qattns'].detach().cpu().numpy()[0]
    question_input_id = question_input['input_ids'].detach().cpu()[0]
    question_tok = tokenizer.convert_ids_to_tokens(question_input_id) 
    def att_softmax(att):
        exps = np.exp(att - np.max(att))
        softmax = exps / np.sum(exps)
        return softmax
    
    img = skimage.io.imread(image_path)

    fig = plt.figure(figsize=(30, 12))
    best_T = 5
    has_and_or = False
    for t in range(T):
        if module_names[np.argmax(np.array(module_prob[t]))] == "And" or module_names[np.argmax(np.array(module_prob[t]))] == "or":
            has_and_or = True
        if t != T-1:
            if module_names[np.argmax(np.array(module_prob[t]))] == "Idle" and module_names[np.argmax(np.array(module_prob[t+1]))] == "Idle":
                best_T = t+1
                break
        else:
            best_T = t+1
            break
    # if not has_and_or:
    #     pass
    # plt.suptitle(,fontsize ='xx-large') 
    # else:
    subfigs = fig.subfigures(1, best_T)

    for t, subfig in enumerate(subfigs.flat):
        subfig.suptitle(f'Time Step @ t = {t}',fontsize ='x-large')
        axs = subfig.subplots(3,1)
        for n, ax in enumerate(axs.flat):
            if n==0:
                    ax.imshow(img)
                    ax.axis('off')
                    ax.set_title('image attention',fontsize ='x-large')
                    mask = att_softmax(image_attns[0][t])
                    mask =skimage.transform.resize(mask,img.shape[:2],order=3)
                    normed_mask = mask / mask.max()
                    normed_mask = (normed_mask * 255).astype('uint8')
                    ax.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap="jet")
                    ax.axis('off')
            if n==1:
                    tem = ax.imshow([np.array(module_prob[t])], cmap='Reds')
                    plt.colorbar(tem,ax=ax,location='bottom')
                    ax.set_xticks(range(len(module_names)), module_names, size='x-small',rotation=30)
                    ax.set_title('module weights',fontsize ='x-large')
                    ax.get_yaxis().set_visible(False)
                    # ax.axis('off')
            if n==2:
                    tem = ax.imshow([question_attn[t]], cmap='Reds')
                    plt.colorbar(tem,ax=ax,location='bottom')
                    ax.set_xticks(range(len(question_tok)), question_tok,size='x-small',rotation=30)
                    ax.set_title('textual attention',fontsize ='x-large')
                    ax.get_yaxis().set_visible(False)
                    # ax.axis('off')
                    
    out_path = str(question)+ '_'+image_path.split('/')[-1]
    save_path =os.path.join(args.vis_dir,out_path)
    plt.savefig(save_path)
    print('visualization saved to ' + save_path)
    plt.close(fig)
    img = Image.open(save_path)
    I1 = ImageDraw.Draw(img)
    font = ImageFont.truetype("/nas-alinlp/lcl193798/lcl_albef_nmn/albef-deepspeed_vqa/arial.ttf", 24)
    I1.text((img.size[0]/2, img.size[1]-int(0.05*img.size[1])), question+f'  prediction: {ans}', font=font, fill=(255, 0, 0)) 
    img.save(save_path)



def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True



    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model #### 
    print("Creating model")

    model = ALBEF(config=config, text_encoder=args.text_encoder, text_decoder=args.text_decoder, tokenizer=tokenizer)
    model = model.to(device)         
  

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
   
    if args.clevr_checkpoint:
        snmn = ['nmn','controller','answer_predictor','init_ctrl']
        # for n,p in model.named_parameters():
        #     if any(nmn in n for nmn in snmn):
        #         print(n)
        #         p.requires_grad = False
        checkpoint = torch.load(args.clevr_checkpoint, map_location='cpu')
        try:
            state_dict = checkpoint['model']
        except:
            state_dict = checkpoint['module']
   
        if not args.evaluate:
            for key in list(state_dict.keys()):
                if not any(nmn in key for nmn in snmn):
                    del state_dict[key]
            print("preparing state dict done!!")
            # for k in state_dict.keys():
            #     print(k)
        msg = model.load_state_dict(state_dict, strict=False)
        print('load nmn checkpoint from %s' % args.clevr_checkpoint)
        # print(msg)

    model_without_ddp = model
    if args.distributed:
        if args.do_amp:
            import apex
            model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    

    if args.stepwise:
        print("Start visualizing")
        image_path = args.image_path
        while True:
            question = input("Ask your question about image:")
            if question =="exist":
                break
            elif question =="switch":
                image_path = input("input new image:")
                question = input("Ask your question about image:")
            visualize_demo(model,question,image_path,tokenizer,device, args)
    else:
        import json
        import tqdm
        if args.task=='clevr':
            imdb = np.load("/nas-alinlp/lcl193798/data_renm/CLEVR/CLEVR_v1.0/imdb/imdb_val.npy", allow_pickle=True)[250:1000]
            for ann in tqdm.tqdm(imdb):
                image_path = ann['image_path']
                question = ann['question_str']    
                visualize_demo(model,question,image_path,tokenizer,device, args)
            
        else:
            # test_file= '/nas-alinlp/lcl193798/albef/data/vqa_ocr_object/vqa_minival_ocr.json'
            test_file ="/nas-alinlp/lcl193798/albef/data/vqa_ocr_object/vqa_test_ocr.json"
            ann = json.load(open(test_file,'r'))[1000:2000]
            vqa_root= '/nas-alinlp/lcl193798/mm_feature/coco_2014/'
            vg_root='/nas-alinlp/lcl193798/mm_feature/visual_genome/'
            gqa_root='/nas-alinlp/lcl193798/mm_feature/'
            for ins in tqdm.tqdm(ann):
                question = ins['question']
                if ins['dataset']=='vqa':
                    image_path = os.path.join(vqa_root,ins['image'])    
                elif ins['dataset']=='vg':
                    image_path = os.path.join(vg_root,ins['image'])  
                elif ins['dataset']=='gqa':
                    image_path = os.path.join(gqa_root,ins['image']) 
                visualize_demo(model,question,image_path,tokenizer,device, args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VQA.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output_dir', default='output/vqa')
    parser.add_argument('--clevr_checkpoint', default='')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--stepwise', action='store_true')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--image_path', default='//nas-alinlp/lcl193798/data_renm/VCR/vcr1images/movieclips_Knowing/RDdc0-JD8Dk@4.jpg')
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
    parser.add_argument('--task', default='vqa', type=str)
    parser.add_argument('--save_raw', action='store_true')  
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
    
    main(args, config)