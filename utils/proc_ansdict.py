# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import sys
sys.path.append('../')
from core.data.ans_punct import prep_ans
import json

DATASET_PATH = '../datasets/vqa/'

ANSWER_PATH = {
    'train': DATASET_PATH + 'v2_mscoco_train2014_annotations.json',
    'val': DATASET_PATH + 'v2_mscoco_val2014_annotations.json',
    'vg': DATASET_PATH + 'VG_annotations.json',
}

# Loading answer word list
stat_ans_list = \
    json.load(open(ANSWER_PATH['train'], 'r'))['annotations'] + \
    json.load(open(ANSWER_PATH['val'], 'r'))['annotations']


def ans_stat(stat_ans_list):
    ans_to_ix = {}
    ix_to_ans = {}
    ans_freq_dict = {}

    for ans in stat_ans_list:
        ans_proc = prep_ans(ans['multiple_choice_answer'])
        if ans_proc not in ans_freq_dict:
            ans_freq_dict[ans_proc] = 1
        else:
            ans_freq_dict[ans_proc] += 1

    ans_freq_filter = ans_freq_dict.copy()
    for ans in ans_freq_dict:
        if ans_freq_dict[ans] <= 8:
            ans_freq_filter.pop(ans)

    for ans in ans_freq_filter:
        ix_to_ans[ans_to_ix.__len__()] = ans
        ans_to_ix[ans] = ans_to_ix.__len__()

    return ans_to_ix, ix_to_ans

ans_to_ix, ix_to_ans = ans_stat(stat_ans_list)
# print(ans_to_ix.__len__())
json.dump([ans_to_ix, ix_to_ans], open('../core/data/answer_dict.json', 'w'))
