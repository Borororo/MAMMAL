import os
import json
import random
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset
from dataset.utils import pre_question
from dataset.preprocess_vqa_answer import VocabDict,extract_answers

import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import re
import numpy as np
import oss2
from io import BytesIO
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

layout_to_id={
    '_NoOp': 0,
    '_Find': 1,
    '_Transform': 2,
    '_Filter': 3,
    '_And': 4,
    '_Describe': 5
    }

class vqa_dataset_vis(Dataset):
    def __init__(self, ann_file, transform, vqa_root, vg_root, gqa_root, eos='[SEP]', split="train", max_ques_words=30, answer_list='', read_local_data=True, add_ocr=False, add_object=False, max_ops = 12, noop_idx = 0,layout_dict=[]):
        self.split = split        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.ann = self.ann[:100]
        self.transform = transform
        self.vqa_root = vqa_root
        self.vg_root = vg_root
        self.gqa_root = gqa_root
        self.max_ques_words = max_ques_words
        self.eos = eos
        self.read_local_data = read_local_data
        self.add_ocr = add_ocr
        self.add_object = add_object

        self.answer_vocab = VocabDict(answer_list,type='albef') 
        self.valid_answer_list = self.answer_vocab.word_list 

        self.max_ops = max_ops
        self.noop_idx = noop_idx
        if split=='test':
            self.max_ques_words = 50 # do not limit question length during test
            self.new_ann = self.ann
        if self.add_ocr:
            self.max_ques_words = 30

        if split=='train':
            self.gt_layout_dict = {}
            for f in layout_dict:
                self.gt_layout_dict.update(np.load(f, allow_pickle=True)[()])
            self.new_ann = self._proprocess_answer(self.ann)
            # acc = 0
            # acccc= 0
            # print(self.new_ann[0])
            # print(list(self.gt_layout_dict.items())[0])
            # for an in self.new_ann:
            #     if an['question_id'] not in  self.gt_layout_dict.keys():
            #         acc+=1
            #     else:
            #         acccc+=1
            # print(acc,acccc)
            
    def _proprocess_answer(self, total_ann):
        new_ann =[]
        unk_ans_count = 0
        for ann in total_ann:
            all_answers, valid_answers, soft_score_inds, soft_score_target = extract_answers(ann['answer'],self.answer_vocab,self.valid_answer_list)
            if len(valid_answers) == 0:
                valid_answers = ['<unk>']
                unk_ans_count += 1
            ann['soft_score_inds'] = soft_score_inds
            ann['soft_score_target'] = soft_score_target
            ann['valid_answers'] = valid_answers
            new_ann.append(ann)
        return new_ann

    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.new_ann[index]
        
        if ann['dataset']=='vqa':
            image_path = os.path.join(self.vqa_root,ann['image'])    
        elif ann['dataset']=='vg':
            image_path = os.path.join(self.vg_root,ann['image'])  
        elif ann['dataset']=='gqa':
            image_path = os.path.join(self.gqa_root,ann['image'])  

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

    
        question = ann['question']
        if self.add_ocr and "ocr" in ann:
            ocrs = ann['ocr']
            ocr_tokens = []
            poses = []
            for ocr in ocrs:
                pos, token = ocr
                ocr_tokens.append(token)
                poses.append(pos)
            if len(ocr_tokens) > 0:
                ocr_string = pre_question(" ".join(ocr_tokens), self.max_ques_words)
                question = question + " [SEP] " + ocr_string
        if self.add_object and "object_label" in ann:
            objects = ann["object_label"]
            question = question + " [SEP] " + " ".join(objects.split("&&"))
        # question = pre_question(question,self.max_ques_words)   
       
        if self.split == 'test':
            question_id = ann['question_id']            
            return image, question, question_id,image_path


        elif self.split=='train':                       
            
            
            if ann['dataset']=='vqa':
                # Soft answer (generation)
                answer_weight = {}
                for answer in ann['answer']:
                        
                    if answer in answer_weight.keys():
                        answer_weight[answer] += 1/len(ann['answer'])
                    else:
                        answer_weight[answer] = 1/len(ann['answer'])

                answers = list(answer_weight.keys())
                weights = list(answer_weight.values())
                # Soft answer (classification)
                soft_socre_inds = ann['soft_score_inds']
                soft_score_target = ann['soft_score_target']
                soft_scores = torch.zeros(self.answer_vocab.num_vocab)
                soft_scores[soft_socre_inds] = torch.tensor(soft_score_target)

                # GT layout
                gt_layout_tokens = self.gt_layout_dict[ann['question_id']]
                for n_t in range(len(gt_layout_tokens) - 1, 0, -1):
                    if (gt_layout_tokens[n_t - 1] in {"_Filter", "_Find"} and gt_layout_tokens[n_t] == "_Filter"):
                        gt_layout_tokens[n_t] = None   
                gt_layout_tokens = [t for t in gt_layout_tokens if t]
                gt_layout_label = torch.tensor([layout_to_id[i] for i in gt_layout_tokens])
                # Pad layout with Noop
                gt_layout_label = torch.nn.functional.pad(gt_layout_label,[0,self.max_ops-len(gt_layout_label)],'constant',self.noop_idx)
                # gt_layout_label = torch.ones(self.max_ops)

            elif ann['dataset']=='vg':
                answers = [ann['answer']]
                weights = [0.5]  
                soft_socre_inds = ann['soft_score_inds']
                soft_score_target = ann['soft_score_target']
                soft_scores = torch.zeros(self.answer_vocab.num_vocab)
                soft_scores[soft_socre_inds] = torch.tensor(soft_score_target)
                gt_layout_label = torch.zeros(self.max_ops)
            elif ann['dataset']=='gqa':
                answers = [ann['answer']]
                weights = [0.5]  
                soft_socre_inds = ann['soft_score_inds']
                soft_score_target = ann['soft_score_target']
                soft_scores = torch.zeros(self.answer_vocab.num_vocab)
                soft_scores[soft_socre_inds] = torch.tensor(soft_score_target)
                gt_layout_label = torch.zeros(self.max_ops)

            answers = [answer+self.eos for answer in answers]
                
            # return image, question, answers, weights
            return image, question, answers, weights, gt_layout_label, soft_scores

