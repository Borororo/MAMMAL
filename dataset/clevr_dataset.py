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




class clevr_dataset_nmn(Dataset):
    def __init__(self, split, config, transform,max_ques_words=30,max_ops = 12, noop_idx = 0, eos='[SEP]'):
        self.split = split        
        if split == 'train':
            self.imdb = np.load(config['train_imdb_file'], allow_pickle=True)
        elif split == 'val':
            self.imdb = np.load(config['val_imdb_file'], allow_pickle=True)
        elif split == 'test':
            self.imdb = np.load(config['test_imdb_file'], allow_pickle=True)
        self.transform = transform
        self.q_vocab_dict = VocabDict(config['vocab_question_file'])
        self.a_vocab_dict = VocabDict(config['vocab_answer_file'])
        self.l_vocab_dict = VocabDict(config['vocab_layout_file'])
        self.prune_filter_module = True
        self.max_ops = max_ops
        self.noop_idx = noop_idx
        self.max_ques_words = max_ques_words
        self.valid_answer_list = self.a_vocab_dict.word_list 
        self.qid = 10000000
        self.eos = eos
    def __len__(self):
        return len(self.imdb)
    
    def __getitem__(self, index):    
        
        ann = self.imdb[index]
        image_path = ann['image_path']    
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          

        question = pre_question(ann['question_str'],self.max_ques_words)   
        question_id = self.qid + index
        answer =[]
        if "answer" in ann and ann["answer"] is not None:
            answer.append(ann['answer'])
            answer_ids = self.a_vocab_dict.word2idx(ann['answer'])
            answer_labels = torch.tensor(answer_ids)
            weights = [0.5]  
        # GT layout
        if self.split == 'test' or self.split == 'val':
            # question_id = ann['question_id']            
            return image, question, question_id,image_path,answer


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
                layout_ids = [self.l_vocab_dict.word2idx(w) for w in gt_layout_tokens]
            gt_layout_label = torch.tensor(layout_ids)
            gt_layout_label = torch.nn.functional.pad(gt_layout_label,[0,self.max_ops-len(gt_layout_label)],'constant',self.noop_idx)

        answer = [ans+self.eos for ans in answer]
        return image, question, question_id,answer,weights, answer_labels, gt_layout_label
    
class clevr_dataset_nmn_visualize(Dataset):
    def __init__(self, split, config, transform,max_ques_words=30,max_ops = 12, noop_idx = 0):
        if split == 'train':
            self.imdb = np.load(config['train_imdb_file'], allow_pickle=True)
        elif split == 'val':
            self.imdb = np.load(config['val_imdb_file'], allow_pickle=True)
        elif split == 'test':
            self.imdb = np.load(config['test_imdb_file'], allow_pickle=True)
        self.transform = transform
        self.q_vocab_dict = VocabDict(config['vocab_question_file'])
        self.a_vocab_dict = VocabDict(config['vocab_answer_file'])
        self.l_vocab_dict = VocabDict(config['vocab_layout_file'])
        self.prune_filter_module = True
        self.max_ops = max_ops
        self.noop_idx = noop_idx
        self.max_ques_words = max_ques_words
        self.valid_answer_list = self.a_vocab_dict.word_list 
        self.qid = 10000000

    def __len__(self):
        return len(self.imdb)
    
    def __getitem__(self, index):    
        
        ann = self.imdb[index]
        image_path = ann['image_path']    
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
 
        question = pre_question(ann['question_str'],self.max_ques_words)   
        question_id = self.qid + index
        answer =[]
        if "answer" in ann and ann["answer"] is not None:
            answer.append(ann['answer'])
            answer_ids = self.a_vocab_dict.word2idx(ann['answer'])
            answer_labels = torch.tensor(answer_ids)
            weights = [0.5]  
        
        # GT layout

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
                layout_ids = [self.l_vocab_dict.word2idx(w) for w in gt_layout_tokens]
            gt_layout_label = torch.tensor(layout_ids)
            gt_layout_label = torch.nn.functional.pad(gt_layout_label,[0,self.max_ops-len(gt_layout_label)],'constant',self.noop_idx)

        return image, question, question_id,answer, weights, answer_labels, gt_layout_label,image_path
