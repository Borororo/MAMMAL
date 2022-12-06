import json
import os
# pred = "//nas-alinlp/lcl193798/lcl_albef_nmn/albef-deepspeed_vqa/output/aokvqa_336BVQA/result/aokvqa_test_result_epoch14.json"

pred="/nas-alinlp/lcl193798/data_renm/vqa_ablef_result/AOKVQA/result/aokvqa_result_epoch0.json"
pred = json.load(open(pred))

norm_pred = {}

for ins in pred:
    norm_pred[list(ins.items())[0][0]] = list(ins.items())[0][1]

writer = open("/nas-alinlp/lcl193798/data_renm/vqa_ablef_result/AOKVQA/result/best_dev_pred.json",'w+')
# writer.write(json.dumps(norm_pred))


ref = '/nas-alinlp/lcl193798/lcl_albef_nmn/albef-deepspeed_vqa/datasets/aokvqa/aokvqa_v1p0_val.json'

ref = json.load(open(ref))

print(len(pred),len(ref))
acc= 0
total =0

coco='/nas-alinlp/lcl193798/lcl_albef_nmn/albef-deepspeed_vqa/datasets/coco/'
def get_coco_path(split, image_id, coco_dir):
    return os.path.join(coco_dir, f"{split}2017", f"{image_id:012}.jpg")


fianl=[]
for ann in ref:
    q_id = ann['question_id']   
    ans = ann['direct_answers']
    image_path  = get_coco_path('val',ann['image_id'],coco)

    pre = norm_pred[str(q_id)]['direct_answer']
    if pre in ans:
        acc+=1
    total+=1
    ann['image_path'] = str(image_path)
    ann["prediction"] = str(pre)
    fianl.append(ann)
writer.write(json.dumps(fianl))
print(f'{acc},{total},{acc/total}')


