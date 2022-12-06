import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset
from dataset.utils import pre_question
import torch


class AnswerTable:
    ANS_CONVERT = {
        "a man": "man",
        "the man": "man",
        "a woman": "woman",
        "the woman": "woman",
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10',
        'grey': 'gray',
    }

    def __init__(self, dsets=None):
        self.all_ans = json.load(open("data/vqa/all_ans.json"))
        if dsets is not None:
            dsets = set(dsets)
            # If the answer is used in the dsets
            self.anss = [ans['ans'] for ans in self.all_ans if
                         len(set(ans['dsets']) & dsets) > 0]
        else:
            self.anss = [ans['ans'] for ans in self.all_ans]
        self.ans_set = set(self.anss)

        self._id2ans_map = self.anss
        self._ans2id_map = {ans: ans_id for ans_id, ans in enumerate(self.anss)}

        assert len(self._id2ans_map) == len(self._ans2id_map)
        for ans_id, ans in enumerate(self._id2ans_map):
            assert self._ans2id_map[ans] == ans_id

    def convert_ans(self, ans):
        if len(ans) == 0:
            return ""
        ans = ans.lower()
        if ans[-1] == '.':
            ans = ans[:-1].strip()
        if ans.startswith("a "):
            ans = ans[2:].strip()
        if ans.startswith("an "):
            ans = ans[3:].strip()
        if ans.startswith("the "):
            ans = ans[4:].strip()
        if ans in self.ANS_CONVERT:
            ans = self.ANS_CONVERT[ans]
        return ans

    def ans2id(self, ans):
        return self._ans2id_map[ans]

    def id2ans(self, ans_id):
        return self._id2ans_map[ans_id]

    def ans2id_map(self):
        return self._ans2id_map.copy()

    def id2ans_map(self):
        return self._id2ans_map.copy()

    def used(self, ans):
        return ans in self.ans_set

    def all_answers(self):
        return self.anss.copy()

    @property
    def num_answers(self):
        return len(self.anss)

class vqa_dataset_cls(Dataset):
    def __init__(self, ann_file, transform, vqa_root, vg_root, eos='[SEP]', split="train", max_ques_words=30,
                 answer_list=''):
        self.split = split
        self.data = []
        for f in ann_file:
            self.data += json.load(open(f, 'r'))

        self.ans2label = json.load(open("data/vqa/trainval_ans2label.json"))
        self.label2ans = json.load(open("data/vqa/trainval_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)
        self.num_answers = len(self.ans2label)

        self.answer_table = AnswerTable()
        for s in self.data:
            if "label" in s:
                label = s["label"]
                for ans in list(label.keys()):
                    new_ans = self.answer_table.convert_ans(ans)
                    if new_ans in self.ans2label:
                        if ans != new_ans:
                            label[new_ans] = label.pop(ans)
                    else:
                        label.pop(ans)
                    if label is None:
                        self.data.remove(s)

        self.transform = transform
        self.vqa_root = vqa_root
        self.vg_root = vg_root
        self.max_ques_words = max_ques_words

        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }
        # self.eos = eos

        # if split == 'test':
        #     self.max_ques_words = 50  # do not limit question length during test
        #     self.answer_list = json.load(open(answer_list, 'r'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datum = self.data[index]
        ques_id = datum['question_id']
        ques = pre_question(datum['sent'], self.max_ques_words)
        img_id = datum['img_id']
        if "val2014" in img_id:
            image_path = os.path.join(self.vqa_root, "val2014/"+img_id+".jpg")
        elif "train2014" in img_id:
            image_path = os.path.join(self.vqa_root, "train2014/"+img_id+".jpg")
        elif "test2015" in img_id:
            image_path = os.path.join(self.vqa_root, "test2015/"+img_id+".jpg")
        else:
            image_path = os.path.join(self.vg_root, "image/"+img_id+".jpg")

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.num_answers)
            for ans, score in label.items():
                target[self.ans2label[ans]] = score
            return ques_id, image, ques, target
        else:
            return ques_id, image, ques

        # if self.split == 'test':
        #     question = pre_question(ann['question'], self.max_ques_words)
        #     question_id = ann['question_id']
        #     return image, question, question_id
        #
        #
        # elif self.split == 'train':
        #
        #     question = pre_question(ann['question'], self.max_ques_words)
        #
        #     if ann['dataset'] == 'vqa':
        #
        #         answer_weight = {}
        #         for answer in ann['answer']:
        #             if answer in answer_weight.keys():
        #                 answer_weight[answer] += 1 / len(ann['answer'])
        #             else:
        #                 answer_weight[answer] = 1 / len(ann['answer'])
        #
        #         answers = list(answer_weight.keys())
        #         weights = list(answer_weight.values())
        #
        #     elif ann['dataset'] == 'vg':
        #         answers = [ann['answer']]
        #         weights = [0.5]
        #
        #     answers = [answer + self.eos for answer in answers]
        #
        #     return image, question, answers, weights
    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)