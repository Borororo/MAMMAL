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

# module_names: ['_NoOp',
#              '_Find',   
#              '_Transform',
#              '_Filter',
#              '_And',
#              '_Describe']

class unify_dataset(Dataset):
    def __init__(self, config, vqa_file, transform, vqa_root, vg_root, eos='[SEP]', split="train", max_ques_words=30, answer_list='', add_ocr=False, add_object=False, max_ops = 12, noop_idx = 0,layout_dict=[]):
        self.split = split        
        self.ann = []
        for f in vqa_file:
            self.ann += json.load(open(f,'r'))
        if split=='train':
            self.gt_layout_dict = {}
            for f in layout_dict:
                self.gt_layout_dict.update(np.load(f, allow_pickle=True)[()])
            
            # print("normalized program with other workds")
            # self.ann = self._normalize_program(self.ann, self.gt_layout_dict)

            # load clevr
            self.imdb = np.load(config['train_imdb_file'], allow_pickle=True)
            self.imdb = self._proprocess_clevr(self.imdb)
            self.ann.extend(self.imdb)
            print('combine clevr and vqa train files')        

        self.transform = transform
        self.vqa_root = vqa_root
        self.vg_root = vg_root
        self.max_ques_words = max_ques_words
        self.eos = eos
        self.add_ocr = add_ocr
        self.add_object = add_object
        self.prune_filter_module = True

        self.answer_vocab = VocabDict(answer_list,type='albef') 
        self.valid_answer_list = self.answer_vocab.word_list 

        self.max_ops = max_ops
        self.noop_idx = noop_idx
        if split=='test':
            self.max_ques_words = 50 # do not limit question length during test
        if self.add_ocr:
            self.max_ques_words = 30
        self.qid = 10000000
    '''
    clevr: 
        image_name, image_path, feature_path, question_str, question_tokens, bbox, answer,gt_layout_tokens

    vqa:
        question_id, question, answer, dataset, ocr, image, object_label, program
    '''
    def _normalize_program(self,vqa_ann,layout_dict):
        for ind in range(len(vqa_ann)):
            if 'program' in vqa_ann[ind]:
                program = vqa_ann[ind]['program']
                if '_Scene' in vqa_ann[ind]['program']:
                    continue
                else:
                    try:
                        program = layout_dict[vqa_ann[ind]['question_id']]
                        for p_t in range(len(program)):
                            if program[p_t] == "_Describe":
                                program[p_t] == "_DescribeOne"
                    except:
                        pass

                    vqa_ann[ind]['program'] = program
            else:
                # print(vqa_ann[ind])
                program = layout_dict[vqa_ann[ind]['question_id']]
                for p_t in range(len(program)):
                    if program[p_t] == "_Describe":
                        program[p_t] == "_DescribeOne"
                vqa_ann[ind]['program'] = program

        
        return vqa_ann
   

    def _proprocess_clevr(self, imdb):
        for ind in range(len(imdb)):
            imdb[ind]['dataset'] = 'clevr'
        return imdb   

    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        # print(ann)
        if ann['dataset']=='vqa':
            image_path = os.path.join(self.vqa_root,ann['image'])    
            question = ann['question']
        elif ann['dataset']=='vg':
            image_path = os.path.join(self.vg_root,ann['image'])  
            question = ann['question']
        elif ann['dataset']=='clevr':
            image_path = ann['image_path']
            question = ann['question_str']
            

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        
        if ann['dataset']=='clevr':
            question = pre_question(ann['question_str'],self.max_ques_words)  
            question_id = self.qid + index
            answer =[]
            if "answer" in ann and ann["answer"] is not None:
                answer.append(ann['answer'])
                weights = [0.5]


            if "gt_layout_tokens" in ann and ann['gt_layout_tokens'] is not None:
                gt_layout_tokens = ann['gt_layout_tokens']
                if self.prune_filter_module:
                    for n_t in range(len(gt_layout_tokens) - 1, 0, -1):
                        if (
                                gt_layout_tokens[n_t - 1] in {"_Filter", "_Find"}
                                and gt_layout_tokens[n_t] == "_Filter"
                        ):
                            gt_layout_tokens[n_t] = None
                gt_layout_tokens = [t for t in gt_layout_tokens if t]
                gt_layout_label = torch.tensor([layout_to_id[i] for i in gt_layout_tokens])
                gt_layout_label = torch.nn.functional.pad(gt_layout_label,[0,self.max_ops-len(gt_layout_label)],'constant',self.noop_idx)

            answer = [ans+self.eos for ans in answer]
            return image, question, answer, weights, gt_layout_label

        else:
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

        
            if self.split == 'test':
                question_id = ann['question_id']            
                return image, question, question_id


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

                    # GT layout
                    try:
                        # print(ann['program'])
                        gt_layout_tokens = ann['program']
                    except:
                        gt_layout_tokens = self.gt_layout_dict[ann['question_id']]

                    for n_t in range(len(gt_layout_tokens) - 1, 0, -1):
                        if (gt_layout_tokens[n_t - 1] in {"_Filter", "_Find"} and gt_layout_tokens[n_t] == "_Filter"):
                            gt_layout_tokens[n_t] = None
                        if  gt_layout_tokens[n_t] == "_Describe":
                            gt_layout_tokens[n_t] = "_DescribeOne"
                    gt_layout_tokens = [t for t in gt_layout_tokens if t]
                    gt_layout_label = torch.tensor([layout_to_id[i] for i in gt_layout_tokens])
                    # Pad layout with Noop
                    gt_layout_label = torch.nn.functional.pad(gt_layout_label,[0,self.max_ops-len(gt_layout_label)],'constant',self.noop_idx)
                    # gt_layout_label = torch.ones(self.max_ops)

                elif ann['dataset']=='vg':
                    answers = [ann['answer']]
                    weights = [0.5]  
                    try:
                        gt_layout_tokens = ann['program']
                        for n_t in range(len(gt_layout_tokens) - 1, 0, -1):
                            if (gt_layout_tokens[n_t - 1] in {"_Filter", "_Find"} and gt_layout_tokens[n_t] == "_Filter"):
                                gt_layout_tokens[n_t] = None
                            if  gt_layout_tokens[n_t] == "_Describe":
                                gt_layout_tokens[n_t] = "_DescribeOne"
                        gt_layout_tokens = [t for t in gt_layout_tokens if t]
                        gt_layout_label = torch.tensor([layout_to_id[i] for i in gt_layout_tokens])
                        # Pad layout with Noop
                        gt_layout_label = torch.nn.functional.pad(gt_layout_label,[0,self.max_ops-len(gt_layout_label)],'constant',self.noop_idx)
                    except:
                        print("warining, no layout labels")
                        gt_layout_label = torch.zeros(self.max_ops)

                answers = [answer+self.eos for answer in answers]
                    
                # return image, question, answers, weights
                return image, question, answers, weights, gt_layout_label

