import json
import sys 

out = sys.argv[1]

# out ="/nas-alinlp/lcl193798/data_renm/vqa_ablef_result/CLEVR_human_zeroshot_vqaN_open/result/clevr_result_epoch0.json"

output =json.load(open(out))


norm={
    "zero":0,
    "one":1,
    "two":2,
    "three":3,
    "four":4,
    "five":5,
    "six":6,
    "seven":7,
    "eight":8,
    "nine":9,
    "ten":10
}

def normal_answer(pred,norm):
    if pred in list(norm.keys()):
        return norm[pred]
    else:
        return pred

correct = 0
total = 0
for ins in output:
    ref = ins['ref']
    pred = str(normal_answer(ins['pred'],norm))
    if ref == pred:
        correct+=1
    total+=1

print(f'acc{correct/total}')
