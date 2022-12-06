import numpy as np
import json
import os
import re
import argparse

import sys
from tqdm import tqdm
from collections import Counter

def load_str_list(fname):
    with open(fname) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines

class VocabDict:
    def __init__(self, vocab_file,type='nmn'):
        if type == 'nmn':
            self.word_list = load_str_list(vocab_file)
        elif type == 'albef':
            self.word_list = json.load(open(vocab_file,'r'))  
        self.word2idx_dict = {w: n_w for n_w, w in enumerate(self.word_list)}
        self.num_vocab = len(self.word_list)
        self.UNK_idx = (
            self.word2idx_dict['<unk>'] if '<unk>' in self.word2idx_dict
            else None)

    def idx2word(self, n_w):
        return self.word_list[n_w]

    def word2idx(self, w):
        if w in self.word2idx_dict:
            return self.word2idx_dict[w]
        elif self.UNK_idx is not None:
            return self.UNK_idx
        else:
            raise ValueError('word %s not in dictionary (while dictionary does'
                             ' not contain <unk>)' % w)

    def tokenize_and_index(self, sentence):
        inds = [self.word2idx(w) for w in tokenize(sentence)]
        return inds



def extract_answers(q_answers,vocab_dict,valid_list):
    all_answers = [answer for answer in q_answers]
    valid_answers = [a for a in all_answers if a in valid_list]
    # build soft scores
    soft_score_inds = []
    soft_score_target = []
    valid_answer_counter = Counter(valid_answers)
    for k, v in valid_answer_counter.items():
        soft_score_inds.append(vocab_dict.word2idx(k))
        soft_score_target.append(min(1., v / 3.))
        # soft_score_target.append(min(1., 0.3*v))
    return all_answers, valid_answers, soft_score_inds, soft_score_target


