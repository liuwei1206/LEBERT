# -*- coding: utf-8 -*-
# @Time    : 2020/11/30 12:47
# @Author  : liuwei
# @File    : to_json.py

import os
from function.format_convert import BMES_to_json
from function.preprocess import get_all_labels_from_corpus

def convert(task_name, mode="train"):
    """
    Args:
        task_name: NER/weibo, NER/note4, POS/CTB5 and so on
        mode: train, dev, or test
    Return:
        json format dataset
    """
    in_file = os.path.join(task_name, mode + ".char.bmes")
    in_file = os.path.join("data/dataset", in_file)
    out_file = os.path.join(task_name, mode + ".json")
    out_file = os.path.join("data/dataset", out_file)

    BMES_to_json(in_file, out_file)

if __name__ == "__main__":
    task_name = "NER/weibo"
    convert(task_name, "train")
    convert(task_name, "dev")
    convert(task_name, "test")
