import json
import numpy as np

train_file= ['/nas-alinlp/lcl193798/albef/data/vqa_ocr_object/vqa_train_ocr_program.json',
             '/nas-alinlp/lcl193798/albef/data/vqa_ocr_object/vqa_nominival_ocr_program.json',
             '/nas-alinlp/lcl193798/albef/data/vg_qa_program.json',
             '/nas-alinlp/lcl193798/albef/data/vqa_ocr_object/vqa_minival_ocr.json']

layout_file= ['/nas-alinlp/lcl193798/data_renm/mcren/Multi-Modal/VQA/stacknmn_v2/vqa/data_process/v2_gt_layout_train2014_new_parse.npy',
             '/nas-alinlp/lcl193798/data_renm/mcren/Multi-Modal/VQA/stacknmn_v2/vqa/data_process/v2_gt_layout_val2014_new_parse.npy']    
data =[]
for f in train_file:
    data += json.load(open(f,'r'))

program_length = []

gt_layout_dict = {}
for f in layout_file:
    gt_layout_dict.update(np.load(f, allow_pickle=True)[()])

for ann in data:
    try:
        gt_layout_tokens = ann['program']
    except:
        gt_layout_tokens = gt_layout_dict[ann['question_id']]
    
    program_length.append(len(gt_layout_tokens))

    if len(gt_layout_tokens) > 8:
        print(ann)



