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


def pre_string(objects,sentence_tokens):
    temp_sentence_tokens =[]
    for tok in sentence_tokens:
        if isinstance(tok,list):
            for ind,i in enumerate(tok):
                if len(tok)==1:
                    temp_sentence_tokens.append(objects[i]+'_'+str(i))
                    continue
                elif ind ==len(tok)-1:
                    temp_sentence_tokens.append('and')
                    temp_sentence_tokens.append(objects[i]+'_'+str(i))
                elif ind != 0 and len(tok)>1:
                    temp_sentence_tokens.append(',')
                    temp_sentence_tokens.append(objects[i]+'_'+str(i))
                else:
                    temp_sentence_tokens.append(objects[i]+'_'+str(i))
        else:
            temp_sentence_tokens.append(tok)
    return ' '.join(temp_sentence_tokens)


likelihood_to_weight={
    "likely":[0.75],
    "possible":[0.5],
    "unlikely":[0.25]
}

def convert_interation_to_weight(answer_weight, iter):
    weight =[]
    for i in iter:
        if i ==0:
            weight.append(answer_weight*1.0)
        if i ==1:
            weight.append(answer_weight*0.5)
        if i ==2:
            weight.append(answer_weight*0.25)
        if i ==3:
            weight.append(answer_weight*0.05)
    return weight


class vcr_dataset(Dataset):
    def __init__(self, split, ann_file,config, transform,max_ques_words=30,max_ops = 12, noop_idx = 0, eos='[SEP]'):
        self.split = split        
        self.ann = []
        for f in ann_file:
            with open(f,'r') as data:
                self.ann += [json.loads(s) for s in data]
        self.transform = transform
        self.vcr_root = config['vcr_root']
        self.max_ques_words = max_ques_words
        self.eos = eos
        self.max_ops = max_ops
        self.noop_idx = noop_idx

    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        ann = self.ann[index]
        image_path = os.path.join(self.vcr_root,ann['img_fn'])  
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        objects = ann['objects']

        question = pre_string(objects,ann['question'])
        
        answer_choices=[
            pre_string(objects,i) for i in ann['answer_choices']
        ]

        rationale_choices = [
            pre_string(objects,i) for i in ann['rationale_choices']
        ]

        question_id = ann['annot_id']   

        if self.split == 'test':
            return image, question, answer_choices, rationale_choices,question_id

   
        elif self.split=='train' or self.split=='val':
            answer_weight = likelihood_to_weight[ann['answer_likelihood']]     
            answer_weights = convert_interation_to_weight(answer_weight[0],ann['answer_match_iter'])                   
            answer = [answer_choices[ann['answer_label']]]
            rationale_weight = [0.5]
            rationale_weights = convert_interation_to_weight(rationale_weight[0],ann['rationale_match_iter'])  
            rationale = [rationale_choices[ann['rationale_label']]]
            if self.split =="val":
                return  image, question, answer,rationale, answer_choices, rationale_choices,question_id
            else:
                return image, question, answer,rationale, answer_choices, rationale_choices,answer_weight,rationale_weight,answer_weights,rationale_weights


class vcr_full_dataset(Dataset):
    def __init__(self, split, ann_file,config, transform,max_ques_words=30,max_ops = 12, noop_idx = 0, eos='[SEP]'):
        self.split = split        
        self.ann = []
        for f in ann_file:
            with open(f,'r') as data:
                self.ann += [json.loads(s) for s in data]
        self.transform = transform
        self.vcr_root = config['vcr_root']
        self.max_ques_words = max_ques_words
        self.eos = eos
        self.max_ops = max_ops
        self.noop_idx = noop_idx

    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        ann = self.ann[index]
        image_path = os.path.join(self.vcr_root,ann['img_fn'])  
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        objects = ann['objects']

        question = pre_string(objects,ann['question'])
        
        answer_choices=[
            pre_string(objects,i) for i in ann['answer_choices']
        ]

        rationale_choices = [
            pre_string(objects,i) for i in ann['rationale_choices']
        ]

        question_id = ann['annot_id']   

        if self.split == 'test':
            return image, question, answer_choices, rationale_choices,question_id

   
        elif self.split=='train' or self.split=='val':
            correct_answer_weights = likelihood_to_weight[ann['answer_likelihood']]
                        
            answer = [answer_choices[ann['answer_label']]]
            rationale_weights = [0.5]
            rationale_weights = convert_interation_to_weight(rationale_weights[0],rationale_choices)  
            rationale = [rationale_choices[ann['rationale_label']]]
            if self.split =="val":
                return  image, question, answer,rationale, answer_choices, rationale_choices,question_id
            else:
                return image, question, answer,rationale, answer_choices, rationale_choices,answer_weights,rationale_weights
