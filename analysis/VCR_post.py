import json
import os
from rouge_score import rouge_scorer
from bert_score  import score
import torch
val_file="//nas-alinlp/lcl193798/data_renm/VCR/val.jsonl"

prediction = "//nas-alinlp/lcl193798/data_renm/analysis/VCR_QA_392_nmn/result/vcr_result_epoch0.json"

prediction = json.load(open(prediction,'r'))
predictions ={}
for r in prediction:
    predictions[r['questionId']]=r['prediction']

def pre_string(objects,sentence_tokens):
    temp_sentence_tokens =[]
    for tok in sentence_tokens:
        if isinstance(tok,list):
            for ind,i in enumerate(tok):
                if len(tok)==1:
                    temp_sentence_tokens.append(objects[i]+' _ '+str(i))
                    continue
                elif ind ==len(tok)-1:
                    temp_sentence_tokens.append('and')
                    temp_sentence_tokens.append(objects[i]+' _ '+str(i))
                elif ind != 0 and len(tok)>1:
                    temp_sentence_tokens.append(',')
                    temp_sentence_tokens.append(objects[i]+' _ '+str(i))
                else:
                    temp_sentence_tokens.append(objects[i]+' _ '+str(i))
        else:
            temp_sentence_tokens.append(tok)
    return ' '.join(temp_sentence_tokens)

# writer_pred = open("/nas-alinlp/lcl193798/data_renm/vqa_ablef_result/VCR_384/pred.txt",'w+')
# writer_ref = open("/nas-alinlp/lcl193798/data_renm/vqa_ablef_result/VCR_384/ref.txt",'w+')

reference = []
with open(val_file,'r') as data:
    reference += [json.loads(s) for s in data]


# rouge1_P= 0.0
# rouge1_R= 0.0
# rouge1_F= 0.0
# rougeL_P= 0.0
# rougeL_R= 0.0
# rougeL_F= 0.0
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

total = 0
acc = 0

cands =[]
refs =[]
labels =[]
rouge =[]
print(len(prediction),len(reference))
for ref in reference:
    current_pred = predictions[ref['annot_id']]

    objects = ref['objects']
    answer_choices=[
        pre_string(objects,i) for i in ref['answer_choices']
    ]
    label = ref['answer_label']
    current_ref = answer_choices[ref['answer_label']]

    # current scores=[]
    for ind, ans_ref  in enumerate(answer_choices):
        cands.append(current_pred)
        refs.append(ans_ref)
        scores = scorer.score(ans_ref, current_pred)
        rouge.append(scores['rougeL'].fmeasure)


    labels.append(label)


P,R,F1 = score(cands,refs,lang='en',verbose=True)
labels = torch.tensor(labels)
rouge=torch.tensor(rouge)
rouge = rouge.view(-1,4)

correct_rouge = torch.count_nonzero(rouge.argmax(dim=-1)==labels)

F1 = F1.view(-1,4)

correct = torch.count_nonzero(F1.argmax(dim=-1)==labels)
total = len(labels)

print("acc:",correct/total)
print("acc_rouge",correct_rouge/total)


        # rouge1_P+=scores['rouge1'].precision
        # rouge1_R+=scores['rouge1'].recall
        # rouge1_F +=scores['rouge1'].fmeasure
        # rougeL_P+=scores['rougeL'].precision
        # rougeL_R+=scores['rougeL'].recall
        # rougeL_F +=scores['rougeL'].fmeasure





