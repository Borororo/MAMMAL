import os
import json
GQA_A = "//nas-alinlp/lcl193798/data_renm/analysis/GQA_layout_halfa/result/gqa_result_epoch0.json"
GQA_B = "//nas-alinlp/lcl193798/data_renm/analysis/GQA_layout_halfb/result/gqa_result_epoch0.json"

half_A = json.load(open(GQA_A))
half_B = json.load(open(GQA_B))
total = half_A+half_B

final ={}
for ins in total:
    final[ins['questionId']] = ins['prediction']


writer = open("//nas-alinlp/lcl193798/data_renm/CLIP_result/GQA_layout_base.json",'w+')

final_json = []

for k,v in final.items():
    final_json.append({
         "questionId": k, 
         "prediction": v
    })
writer.write(json.dumps(final_json))
writer.close()
