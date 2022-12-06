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


def load_aokvqa(aokvqa_dir, split, version='v1p0'):
    assert split in ['train', 'val', 'test', 'test_w_ans']
    dataset = json.load(open(
        os.path.join(aokvqa_dir, f"aokvqa_{version}_{split}.json")
    ))
    return dataset

def get_coco_path(split, image_id, coco_dir):
    return os.path.join(coco_dir, f"{split}2017", f"{image_id:012}.jpg")


def get_ok_coco_path(img_id,coco_dir="/nas-alinlp/lcl193798/mm_feature/coco_2014/"):
    img_path_train = os.path.join("/nas-alinlp/lcl193798/mm_feature/coco_2014/", f"train2014_img", f"COCO_train2014_{img_id:012}.jpg")
    img_path_val = os.path.join("/nas-alinlp/lcl193798/mm_feature/coco_2014/", f"val2014_img", f"COCO_val2014_{img_id:012}.jpg")
    if os.path.exists(img_path_train):
        return img_path_train
    else:
        return img_path_val

class aokvqa_dataset(Dataset):
    def __init__(self, ann_file, transform, coco_root, eos='[SEP]', split="train", max_ques_words=30):
        self.split = split        
        self.ann = []
        for f in ann_file:
            self.ann += load_aokvqa(f,split)
        # if split == 'train':
        #     print("combining OKVQA!!!")
        #     dataset = json.load(open(os.path.join(ann_file[0], "okvqa_train.json")))
        #     self.ann.extend(dataset)
        # self.ann = self.ann[:100]
        self.transform = transform
        self.coco_root = coco_root
        self.max_ques_words = max_ques_words
        self.eos = eos


    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        try:
            image_path  = get_coco_path(self.split,ann['image_id'],self.coco_root)
            image = Image.open(image_path).convert('RGB')
        except:
            image_path = get_ok_coco_path(ann['image_id'])
            image = Image.open(image_path).convert('RGB')
        image = self.transform(image)        
        question = ann['question']
       
        if self.split == 'test' or self.split == 'val':
            question_id = ann['question_id']            
            return image, question, question_id

        else:
            # Soft answer (generation)
            answer_weight = {}
            for answer in ann['direct_answers']:
                if answer in answer_weight.keys():
                    answer_weight[answer] += 1/len(ann['direct_answers'])
                else:
                    answer_weight[answer] = 1/len(ann['direct_answers'])

            answers = list(answer_weight.keys())
            weights = list(answer_weight.values())

            answers = [answer+self.eos for answer in answers]
            # rationales = ' '.join(ann['rationales'])
            # question = question +' ' +rationales
            return image, question, answers, weights

