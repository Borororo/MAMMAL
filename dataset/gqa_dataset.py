import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
import re
import json
import os
import numpy as np
from dataset.preprocess_vqa_answer import VocabDict,extract_answers

layout_to_id = {
    "_NoOp": 0,
    "_Find": 1,
    "_Transform": 2,
    "_Filter": 3,
    "_And": 4,
    "_Or": 5,
    "_Scene": 6,
    "_DescribeOne": 7,
    "_DescribeTwo": 8,
}
# preprocess question
def pre_question(question,max_ques_words):
    question = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        question.lower(),
    ).replace('-', ' ').replace('/', ' ')  
    question = question.rstrip(' ')
    
    #truncate question
    question_words = question.split(' ')
    if len(question_words)>max_ques_words:
        question = ' '.join(question_words[:max_ques_words])
            
    return question




class gqa_dataset_nmn(Dataset):
    def __init__(self, split, ann_file,config, transform,max_ques_words=30,max_ops = 12, noop_idx = 0, eos='[SEP]'):
        self.split = split        
        self.ann = []
        if split =='train' or split =="val":
            for f in ann_file:
                self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.gqa_root = config['gqa_root']
        self.suffix = config["gqa_suffix"]
        self.max_ques_words = max_ques_words
        self.eos = eos
        self.max_ops = max_ops
        self.noop_idx = noop_idx
        if split=='test':
            self.new_ann=[]
            self.ann=json.load(open(ann_file[0],'r'))
            for k,v in self.ann.items():
                v['question_id'] = k
                self.new_ann.append(v)
            max_len = len(self.new_ann)
            self.ann = self.new_ann[:int(max_len/2)]
            # self.ann = self.new_ann[int(max_len/2)-5:]
            

    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        ann = self.ann[index]
        if self.split == 'test':
            image_path = os.path.join(self.gqa_root,ann['imageId']+self.suffix)  
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            question = ann['question']
            question_id = ann['question_id']            
            return image, question, question_id

        
        elif self.split=='train' or self.split=='val':                       
            image_path = os.path.join(self.gqa_root,ann['imageid']+self.suffix)  
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            question = ann['question']
            answers = [ann['answer']]
            weights = [0.5]  
            answers = [answer+self.eos for answer in answers]
            gt_layout_tokens = ann['norm_program'].split('->')
            for n_t in range(len(gt_layout_tokens) - 1, 0, -1):
                if (gt_layout_tokens[n_t - 1] in {"_Filter", "_Find"} and gt_layout_tokens[n_t] == "_Filter"):
                    gt_layout_tokens[n_t] = None
                if  gt_layout_tokens[n_t] == "_Describe":
                    gt_layout_tokens[n_t] = "_DescribeOne"
            gt_layout_tokens = [t for t in gt_layout_tokens if t]
            gt_layout_label = torch.tensor([layout_to_id[i] for i in gt_layout_tokens])

            gt_layout_label = torch.nn.functional.pad(gt_layout_label,[0,self.max_ops-len(gt_layout_label)],'constant',self.noop_idx)

            return image, question, answers, weights, gt_layout_label

