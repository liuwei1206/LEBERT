# -*- coding: utf-8 -*-
# @Time    : 2020/12/01 11:47
# @Author  : liuwei
# @File    : format_convert.py

"""
convert data format
"""
import json
import os
import time
from tqdm import tqdm, trange

def BMES_to_json(bmes_file, json_file):
    """
    convert bmes format file to json file, json file has two key, including text and label
    Args:
        bmes_file:
        json_file:
    :return:
    """
    texts = []
    with open(bmes_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        total_line_num = len(lines)
        line_iter = trange(total_line_num)
        words = []
        labels = []
        for idx in line_iter:
            line = lines[idx]
            line = line.strip()

            if not line:
                assert len(words) == len(labels), (len(words), len(labels))
                sample = {}
                sample['text'] = words
                sample['label'] = labels
                texts.append(json.dumps(sample, ensure_ascii=False))

                words = []
                labels = []
            else:
                items = line.split()
                words.append(items[0])
                labels.append(items[1])

    with open(json_file, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write("%s\n"%(text))
