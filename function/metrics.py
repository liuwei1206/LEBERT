# -*- coding: utf-8 -*-
# @Time    : 2020/12/01 11:47
# @Author  : liuwei
# @File    : metrics.py
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score

def seq_f1_with_mask(all_true_labels, all_pred_labels, all_label_mask, label_vocab):
    """
    For Chinese, since label is given to each character, do not exists subtoken,
    so we can evaluate in character level directly, extra processing

    Args:
        all_true_labels: true label ids
        all_pred_labels: predict label ids
        all_label_mask: the valid of each position
        label_vocab: from id to labels
    """
    assert len(all_true_labels) == len(all_pred_labels), (len(all_true_labels), len(all_pred_labels))
    assert len(all_true_labels) == len(all_label_mask), (len(all_true_labels), len(all_label_mask))

    true_labels = []
    pred_labels = []

    sample_num = len(all_true_labels)
    for i in range(sample_num):
        tmp_true = []
        tmp_pred = []

        assert len(all_true_labels[i]) == len(all_pred_labels[i]), (len(all_true_labels[i]), len(all_pred_labels[i]))
        assert len(all_true_labels[i]) == len(all_label_mask[i]), (len(all_true_labels[i]), len(all_label_mask[i]))

        real_seq_length = np.sum(all_label_mask[i])
        for j in range(1, real_seq_length-1): # remove the label of [CLS] and [SEP]
            if all_label_mask[i][j] == 1:
                tmp_true.append(label_vocab.convert_id_to_item(all_true_labels[i][j]).replace("M-", "I-"))
                tmp_pred.append(label_vocab.convert_id_to_item(all_pred_labels[i][j]).replace("M-", "I-"))

        true_labels.append(tmp_true)
        pred_labels.append(tmp_pred)

    acc = accuracy_score(true_labels, pred_labels)
    p = precision_score(true_labels, pred_labels)
    r = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    return acc, p, r, f1, true_labels, pred_labels
