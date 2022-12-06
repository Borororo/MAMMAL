import json
import os
import argparse
Q1 =  ['count']

Q2 = [
'equal_color',
'equal_material',
'equal_shape',
'equal_size']

Q3 =  ['exist']

Q4 = [
'greater_than',
'less_than',
'equal_integer']

Q5 = [
'query_color',
'query_material',
'query_shape',
'query_size']


def compute_accuracy(predicts):
    total = 0
    acc = 0
    for ins in predicts:
        if  ins['pred'] == ins['ref']:
            acc+=1
        total +=1
    return acc, total

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file_dir', default='output/vqa')
    parser.add_argument('--output_dir', default='output/vqa')
    args = parser.parse_args()

    refere_dev_question = "//nas-alinlp/lcl193798/data_renm/CLEVR/clevr_only_question/CLEVR_v1.0/questions/CLEVR_val_questions.json"
    
    reference = json.load(open(refere_dev_question))['questions']
    predictions = json.load(open(args.prediction_file_dir,'r'))

    predictions = predictions[:len(reference)]

    count_list = []
    compare_attribute =[]
    exist =[]
    compare_number =[]
    query_attribute =[]
    
    
    for pred,ref in zip(predictions, reference):
        last_function =ref['program'][-1]['function'] 
        if last_function in Q1:
            count_list.append(pred)
        if last_function in Q2:
            compare_attribute.append(pred)
        if last_function in Q3:
            exist.append(pred)
        if last_function in Q4:
            compare_number.append(pred)
        if last_function in Q5:
            query_attribute.append(pred)
        
    
    t_a,t_t = compute_accuracy(predictions)
    q1_a,q1_t = compute_accuracy(count_list)
    q2_a,q2_t = compute_accuracy(compare_attribute)
    q3_a,q3_t = compute_accuracy(exist)
    q4_a,q4_t = compute_accuracy(compare_number)
    q5_a,q5_t = compute_accuracy(query_attribute)

    result = {
        "total": [t_a,t_t,t_a/t_t],
        "count":[q1_a,q1_t,q1_a/q1_t],
        "compare_attribute":[q2_a,q2_t,q2_a/q2_t],
        "exist":[q3_a,q3_t,q3_a/q3_t],
        "compare_number":[q4_a,q4_t,q4_a/q4_t],
        "query_attribute":[q5_a,q5_t,q5_a/q5_t],
    }

    print(result)
    writer = open(args.output_dir,'w+')
    writer.write(json.dumps(result,indent=2))
    writer.close()
