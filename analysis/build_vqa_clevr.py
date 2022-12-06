import json
import numpy as np 


vqa_root= '/nas-alinlp/lcl193798/mm_feature/coco_2014/'
vg_root='/nas-alinlp/lcl193798/mm_feature/visual_genome/'

vqa_train_file= ['/nas-alinlp/lcl193798/albef/data/vqa_ocr_object/vqa_train_ocr_program.json',
             '/nas-alinlp/lcl193798/albef/data/vqa_ocr_object/vqa_nominival_ocr_program.json',
             '/nas-alinlp/lcl193798/albef/data/vg_qa_program.json',
             '/nas-alinlp/lcl193798/albef/data/vqa_ocr_object/vqa_minival_ocr.json']



vqa_layout_file= ['/nas-alinlp/lcl193798/data_renm/mcren/Multi-Modal/VQA/stacknmn_v2/vqa/data_process/v2_gt_layout_train2014_new_parse.npy',
             '/nas-alinlp/lcl193798/data_renm/mcren/Multi-Modal/VQA/stacknmn_v2/vqa/data_process/v2_gt_layout_val2014_new_parse.npy']    


train_imdb_file: "/nas-alinlp/lcl193798/data_renm/CLEVR/CLEVR_v1.0/imdb/imdb_train.npy"
val_imdb_file: "/nas-alinlp/lcl193798/data_renm/CLEVR/CLEVR_v1.0/imdb/imdb_val.npy"
test_imdb_file: "/nas-alinlp/lcl193798/data_renm/CLEVR/CLEVR_v1.0/imdb/imdb_test.npy"