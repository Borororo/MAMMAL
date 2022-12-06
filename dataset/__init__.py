import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset.caption_dataset import re_train_dataset, re_eval_dataset, pretrain_dataset, coco_dataset, ggw_dataset
from dataset.nlvr_dataset import nlvr_dataset
from dataset.ve_dataset import ve_dataset
from dataset.vqa_dataset import vqa_dataset
from dataset.gqa_dataset import gqa_dataset_nmn
from dataset.vqa_dataset_cls import vqa_dataset_cls
from dataset.vqa_dataset_nmn import vqa_dataset_nmn
from dataset.vqa_dataset_nmn_vis import vqa_dataset_vis
from dataset.grounding_dataset import grounding_dataset
from dataset.clevr_dataset import clevr_dataset_nmn,clevr_dataset_nmn_visualize
from dataset.clevr_human_dataset import clevr_human_dataset_nmn
from dataset.vcr_dataset import vcr_dataset
from dataset.aokvqa_dataset import aokvqa_dataset
from dataset.UNIFY_dataset import unify_dataset
from dataset.randaugment import RandomAugment

def create_dataset(dataset, config):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    pretrain_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.2, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])    
    train_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])  
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])   
    
    if dataset=='pretrain':
        dataset = pretrain_dataset(config['train_file'], pretrain_transform, read_local_data=config['read_local_data'])
        return dataset      
               
    elif dataset=='re':          
        train_dataset = re_train_dataset(config['train_file'], train_transform, config['image_root'])
        val_dataset = re_eval_dataset(config['val_file'], test_transform, config['image_root'])  
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])                
        return train_dataset, val_dataset, test_dataset   

    elif dataset=='vqa': 
        train_dataset = vqa_dataset(config['train_file'], train_transform, config['vqa_root'], config['vg_root'], config['gqa_root'], split='train',answer_list=config['answer_list'], read_local_data=config['read_local_data'], add_ocr=config['add_ocr'], add_object=config['add_object']) 
        vqa_test_dataset = vqa_dataset(config['test_file'], test_transform, config['vqa_root'], config['vg_root'], config['gqa_root'], split='test', answer_list=config['answer_list'], read_local_data=config['read_local_data'], add_ocr=config['add_ocr'], add_object=config['add_object'])       
        vqa_val_dataset = vqa_dataset(config['val_file'], test_transform, config['vqa_root'], config['vg_root'], config['gqa_root'],split='test', answer_list=config['answer_list'], read_local_data=config['read_local_data'], add_ocr=config['add_ocr'], add_object=config['add_object'])       
        return train_dataset, vqa_val_dataset, vqa_test_dataset
    elif dataset=='vqa-nmn':
        train_dataset = vqa_dataset_nmn(config['train_file'], train_transform, config['vqa_root'], config['vg_root'], config['gqa_root'], split='train', answer_list=config['answer_list'],read_local_data=config['read_local_data'], add_ocr=config['add_ocr'], add_object=config['add_object'],max_ops = config['controller_nodes'], noop_idx = config['noop_idx'],layout_dict=config['layout_file']) 
        vqa_test_dataset = vqa_dataset_nmn(config['test_file'], test_transform, config['vqa_root'], config['vg_root'], config['gqa_root'], split='test', answer_list=config['answer_list'], read_local_data=config['read_local_data'], add_ocr=config['add_ocr'], add_object=config['add_object'],max_ops = config['controller_nodes'], noop_idx = config['noop_idx'],layout_dict=config['layout_file'])       
        vqa_val_dataset = vqa_dataset_nmn(config['val_file'], test_transform, config['vqa_root'], config['vg_root'], config['gqa_root'],split='test', answer_list=config['answer_list'], read_local_data=config['read_local_data'], add_ocr=config['add_ocr'], add_object=config['add_object'],max_ops = config['controller_nodes'], noop_idx = config['noop_idx'],layout_dict=config['layout_file'])    
        return train_dataset, vqa_val_dataset, vqa_test_dataset
    elif dataset=='vqa-vis':
        # train_dataset = vqa_dataset_nmn(config['train_file'], train_transform, config['vqa_root'], config['vg_root'], config['gqa_root'], split='train', answer_list=config['answer_list'],read_local_data=config['read_local_data'], add_ocr=config['add_ocr'], add_object=config['add_object'],max_ops = config['controller_nodes'], noop_idx = config['noop_idx'],layout_dict=config['layout_file']) 
        vqa_test_dataset = vqa_dataset_vis(config['test_file'], test_transform, config['vqa_root'], config['vg_root'], config['gqa_root'], split='test', answer_list=config['answer_list'], read_local_data=config['read_local_data'], add_ocr=config['add_ocr'], add_object=config['add_object'],max_ops = config['controller_nodes'], noop_idx = config['noop_idx'],layout_dict=config['layout_file'])       
        vqa_val_dataset = vqa_dataset_vis(config['val_file'], test_transform, config['vqa_root'], config['vg_root'], config['gqa_root'],split='test', answer_list=config['answer_list'], read_local_data=config['read_local_data'], add_ocr=config['add_ocr'], add_object=config['add_object'],max_ops = config['controller_nodes'], noop_idx = config['noop_idx'],layout_dict=config['layout_file'])    
        return vqa_val_dataset, vqa_test_dataset
    elif dataset=="clevr":
        train_dataset = clevr_dataset_nmn(split='train', config=config, max_ques_words=30, transform=train_transform,max_ops = config['controller_nodes'], noop_idx = config['noop_idx'])    
        val_dataset = clevr_dataset_nmn(split='val', config=config,max_ques_words=30, transform=test_transform,max_ops = config['controller_nodes'], noop_idx = config['noop_idx'])         
        return train_dataset,val_dataset
    elif dataset=="clevr_human":
        train_dataset = clevr_human_dataset_nmn(split='train', config=config, max_ques_words=30, transform=train_transform,max_ops = config['controller_nodes'], noop_idx = config['noop_idx'])    
        val_dataset = clevr_human_dataset_nmn(split='val', config=config,max_ques_words=30, transform=test_transform,max_ops = config['controller_nodes'], noop_idx = config['noop_idx'])         
        return train_dataset,val_dataset
    elif dataset=="gqa":
        train_dataset = gqa_dataset_nmn(split='train', ann_file= config['train_file'], config=config, max_ques_words=30, transform=train_transform,max_ops = config['controller_nodes'], noop_idx = config['noop_idx'])    
        val_dataset = gqa_dataset_nmn(split='val', ann_file= config['val_file'],config=config,max_ques_words=30, transform=test_transform,max_ops = config['controller_nodes'], noop_idx = config['noop_idx'])
        test_dataset = gqa_dataset_nmn(split='test',ann_file= config['test_file'], config=config,max_ques_words=30, transform=test_transform,max_ops = config['controller_nodes'], noop_idx = config['noop_idx'])             
        return train_dataset,val_dataset,test_dataset
    elif dataset=="vcr":
        train_dataset = vcr_dataset(split='train', ann_file= config['train_file'], config=config, max_ques_words=30, transform=train_transform,max_ops = config['controller_nodes'], noop_idx = config['noop_idx'])    
        val_dataset = vcr_dataset(split='val', ann_file= config['val_file'],config=config,max_ques_words=30, transform=test_transform,max_ops = config['controller_nodes'], noop_idx = config['noop_idx'])
        test_dataset = vcr_dataset(split='test',ann_file= config['test_file'], config=config,max_ques_words=30, transform=test_transform,max_ops = config['controller_nodes'], noop_idx = config['noop_idx'])             
        return train_dataset,val_dataset,test_dataset
    
    elif dataset== 'coco':
        train_dataset = coco_dataset(config['train_file'], train_transform, config['coco_root'], max_words=config['max_length'], read_local_data=config['read_local_data'], is_train=True, add_object=config['add_object'])
        val_dataset = coco_dataset(config['test_file'], test_transform, config['coco_root'], max_words=config['max_length'], read_local_data=config['read_local_data'], is_train=False, add_object=config['add_object'])
        test_dataset = coco_dataset(config['test_file'], test_transform, config['coco_root'], max_words=config['max_length'], read_local_data=config['read_local_data'], is_train=False, add_object=config['add_object'])
        return train_dataset, val_dataset, test_dataset
    elif dataset== 'ggw':
        train_dataset = ggw_dataset(config['train_file'], max_words=config['max_length'], read_local_data=config['read_local_data'], is_train=True)
        val_dataset = ggw_dataset(config['test_file'], max_words=config['max_length'], read_local_data=config['read_local_data'], is_train=False)
        test_dataset = ggw_dataset(config['test_file'], max_words=config['max_length'], read_local_data=config['read_local_data'], is_train=False)
        return train_dataset, val_dataset, test_dataset

    elif dataset == "vqa_cls":
        train_dataset = vqa_dataset_cls(config['train_file'], train_transform, config['vqa_root'], config['vg_root'], split='train')
        val_dataset = vqa_dataset_cls(config['val_file'], train_transform, config['vqa_root'], config['vg_root'],
                                    split='val')
        vqa_test_dataset = vqa_dataset_cls(config['test_file'], test_transform, config['vqa_root'], config['vg_root'], split='test', answer_list=config['answer_list'])
        return train_dataset, val_dataset, vqa_test_dataset
    elif dataset=='nlvr':   
        train_dataset = nlvr_dataset(config['train_file'], train_transform, config['image_root'])  
        val_dataset = nlvr_dataset(config['val_file'], test_transform, config['image_root'])  
        test_dataset = nlvr_dataset(config['test_file'], test_transform, config['image_root'])                
        return train_dataset, val_dataset, test_dataset        
               
    elif dataset=='ve':   
        train_dataset = ve_dataset(config['train_file'], train_transform, config['image_root'])  
        val_dataset = ve_dataset(config['val_file'], test_transform, config['image_root'])  
        test_dataset = ve_dataset(config['test_file'], test_transform, config['image_root'])                
        return train_dataset, val_dataset, test_dataset     
    
    elif dataset=='grounding':
        train_transform = transforms.Compose([                        
                transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])         
        train_dataset = grounding_dataset(config['train_file'], train_transform, config['image_root'], mode='train')       
        test_dataset = grounding_dataset(config['test_file'], test_transform, config['image_root'], mode='test')             
        return train_dataset, test_dataset    
    elif dataset == "aokvqa":
            # def __init__(self, ann_file, transform, coco_root, eos='[SEP]', split="train", max_ques_words=30):
        train_dataset = aokvqa_dataset(config['train_file'], train_transform, config['coco_root'], split='train') 
        test_dataset = aokvqa_dataset(config['test_file'], test_transform, config['coco_root'], split='test')       
        val_dataset = aokvqa_dataset(config['val_file'], test_transform, config['coco_root'],split='val')       
        return train_dataset, val_dataset, test_dataset

    elif dataset == "unify":
            # def __init__(self, vqa_file, transform, vqa_root, vg_root, eos='[SEP]', split="train", max_ques_words=30, answer_list='', add_ocr=False, add_object=False, max_ops = 12, noop_idx = 0,layout_dict=[]):
        train_dataset = unify_dataset(config, config['train_file'], train_transform, config['vqa_root'], config['vg_root'], split='train', answer_list=config['answer_list'], add_ocr=config['add_ocr'], add_object=config['add_object'],max_ops = config['controller_nodes'], noop_idx = config['noop_idx'],layout_dict=config['layout_file']) 
        val_dataset = unify_dataset(config, config['val_file'], train_transform, config['vqa_root'], config['vg_root'], split='val', answer_list=config['answer_list'], add_ocr=config['add_ocr'], add_object=config['add_object'],max_ops = config['controller_nodes'], noop_idx = config['noop_idx'],layout_dict=config['layout_file']) 
        test_dataset = unify_dataset(config, config['test_file'], train_transform, config['vqa_root'], config['vg_root'], split='test', answer_list=config['answer_list'], add_ocr=config['add_ocr'], add_object=config['add_object'],max_ops = config['controller_nodes'], noop_idx = config['noop_idx'],layout_dict=config['layout_file']) 
     
        return train_dataset, val_dataset, test_dataset


def clevr_collate_fn(batch):
    image_list, question_list, answer_list,weight_list,answer_label_list, n, gt_label_list,question_ids= [], [], [], [], [],[],[],[]
    for image, question, question_id, answer, weights,answer_labels, layouts in batch:
        image_list.append(image)
        question_list.append(question)     
        answer_list += answer
        weight_list += weights    
        question_ids.append(question_id)
        gt_label_list.append(layouts)
        n.append(len(answer))
        answer_label_list.append(answer_labels)
    return torch.stack(image_list,dim=0), question_list,question_ids,answer_list,torch.Tensor(weight_list), n,torch.stack(gt_label_list,dim=0),torch.stack(answer_label_list,dim=0)

def vcr_collate_fn(batch):
    image_list, question_list, answer_list, rationale_list, answer_weight_list,rationale_weight_list,n = [], [], [], [], [],[],[]
    
    for image, question, answer,rationale, answer_choices, rationale_choices,answer_weight,rationale_weight,answer_weights,rationale_weights in batch:
        image_list.append(image)
        question_list.append(question)     
        answer_list += answer
        rationale_list += rationale
        answer_weight_list += answer_weight
        rationale_weight_list += rationale_weight
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list,rationale_list, torch.Tensor(answer_weight_list), torch.Tensor(rationale_weight_list),n

def vcr_multi_collate_fn(batch):
    image_list, question_list, answer_list, rationale_list, answer_weight_list,rationale_weight_list,n = [], [], [], [], [],[],[]
    
    for image, question, answer,rationale, answer_choices, rationale_choices,answer_weight,rationale_weight,answer_weights,rationale_weights in batch:
        image_list.append(image)
        question_list.append(question)     
        answer_list += answer_choices
        rationale_list += rationale_choices
        answer_weight_list += answer_weights
        rationale_weight_list += rationale_weights
        n.append(len(answer_choices))
    return torch.stack(image_list,dim=0), question_list, answer_list,rationale_list, torch.Tensor(answer_weight_list), torch.Tensor(rationale_weight_list),n


def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n


def unify_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n,gt_label_list= [], [], [], [], [],[]
    for image, question, answer, weights,layouts in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        gt_label_list.append(layouts)
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n,torch.stack(gt_label_list,dim=0)

def vqa_nmn_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n, gt_label_list,soft_scores= [], [], [], [], [],[],[]
    for image, question, answer, weights, layouts, soft_answers in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        gt_label_list.append(layouts)
        soft_scores.append(soft_answers)
        n.append(len(answer))
    # print(torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n)
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n,torch.stack(gt_label_list,dim=0),torch.stack(soft_scores,dim=0)

def gqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n, gt_label_list= [], [], [], [], [],[]
    for image, question, answer, weights, layouts in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        gt_label_list.append(layouts)
        n.append(len(answer))

    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n,torch.stack(gt_label_list,dim=0)


def coco_collate_fn(batch):
    image_list, caption_list, object_labels, image_id_list, gold_caption_list = [], [], [], [], []
    for image, caption, object_label, image_id, gold_caption in batch:
        image_list.append(image)
        caption_list.append(caption)
        image_id_list.append(image_id)
        gold_caption_list.append(gold_caption)
        object_labels.append(object_label)
    return torch.stack(image_list,dim=0), caption_list, object_labels, image_id_list, gold_caption_list

def ggw_collate_fn(batch):
    encoder_input_list, decoder_input_list = [], []
    for encoder_input, decoder_input in batch:
        encoder_input_list.append(encoder_input)
        decoder_input_list.append(decoder_input)
    return encoder_input_list, decoder_input_list


def vqa_cls_collate_fn(batch):
    if len(batch[0]) == 4:
        ques_id, feats, ques, target = zip(*batch)
    else:
        ques_id, feats, ques = zip(*batch)

    feats = torch.stack(feats, dim=0)
    if len(batch[0]) == 4:
        target = torch.stack(target, dim=0)
        return ques_id, feats, ques, target
    else:
        return ques_id, feats, ques




def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    
