# -*- coding: utf-8 -*-
# @Time    : 2020/11/26 15:47
# @Author  : liuwei
# @File    : preprocess.py

import time
import os
import json
from tqdm import tqdm, trange
from module.lexicon_tree import Trie

def sent_to_matched_words_boundaries(sent, lexicon_tree, max_word_num=None):
    """
    输入一个句子和词典树, 返回句子中每个字所属的匹配词, 以及该字的词边界
    字可能属于以下几种边界:
        B-: 词的开始, 0
        M-: 词的中间, 1
        E-: 词的结尾, 2
        S-: 单字词, 3
        BM-: 既是某个词的开始, 又是某个词中间, 4
        BE-: 既是某个词开始，又是某个词结尾, 5
        ME-: 既是某个词的中间，又是某个词结尾, 6
        BME-: 词的开始、词的中间和词的结尾, 7

    Args:
        sent: 输入的句子, 一个字的数组
        lexicon_tree: 词典树
        max_word_num: 最多匹配的词的数量
    Args:
        sent_words: 句子中每个字归属的词组
        sent_boundaries: 句子中每个字所属的边界类型
    """
    sent_length = len(sent)
    sent_words = [[] for _ in range(sent_length)]
    sent_boundaries = [[] for _ in range(sent_length)]  # each char has a boundary

    for idx in range(sent_length):
        sub_sent = sent[idx:idx + lexicon_tree.max_depth]  # speed using max depth
        words = lexicon_tree.enumerateMatch(sub_sent)

        if len(words) == 0 and len(sent_boundaries[idx]) == 0:
            sent_boundaries[idx].append(3) # S-
        else:
            if len(words) == 1 and len(words[0]) == 1: # single character word
                if len(sent_words[idx]) == 0:
                    sent_words[idx].extend(words)
                    sent_boundaries[idx].append(3) # S-
            else:
                if max_word_num:
                    need_num = max_word_num - len(sent_words[idx])
                    words = words[:need_num]
                sent_words[idx].extend(words)
                for word in words:
                    if 0 not in sent_boundaries[idx]:
                        sent_boundaries[idx].append(0) # S-
                    start_pos = idx + 1
                    end_pos = idx + len(word) - 1
                    for tmp_j in range(start_pos, end_pos):
                        if 1 not in sent_boundaries[tmp_j]:
                            sent_boundaries[tmp_j].append(1) # M-
                        sent_words[tmp_j].append(word)
                    if 2 not in sent_boundaries[end_pos]:
                        sent_boundaries[end_pos].append(2) # E-
                    sent_words[end_pos].append(word)

    assert len(sent_words) == len(sent_boundaries)

    new_sent_boundaries = []
    idx = 0
    for boundary in sent_boundaries:
        if len(boundary) == 0:
            print("Error")
            new_sent_boundaries.append(0)
        elif len(boundary) == 1:
            new_sent_boundaries.append(boundary[0])
        elif len(boundary) == 2:
            total_num = sum(boundary)
            new_sent_boundaries.append(3 + total_num)
        elif len(boundary) == 3:
            new_sent_boundaries.append(7)
        else:
            print(boundary)
            print("Error")
            new_sent_boundaries.append(8)
    assert len(sent_words) == len(new_sent_boundaries)

    return sent_words, new_sent_boundaries

def sent_to_distinct_matched_words(sent, lexicon_tree):
    """
    得到句子的匹配词, 并进行分组, 按照BMES进行分组
    Args:
        sent: 一个字的数组
        lexicon_tree: 词汇表树
        max_word_num: 最大词数
    """
    sent_length = len(sent)
    sent_words = [[[], [], [], []] for _ in range(sent_length)] # 每个字都有对应BMES
    sent_group_mask = [[0, 0, 0, 0] for _ in range(sent_length)]

    for idx in range(sent_length):
        sub_sent = sent[idx:idx+lexicon_tree.max_depth]
        words = lexicon_tree.enumerateMatch(sub_sent)
        if len(words) == 0:
            continue
        else:
            for word in words:
                word_length = len(word)
                if word_length == 1:
                    sent_words[idx][3].append(word)
                    sent_group_mask[idx][3] = 1
                else:
                    sent_words[idx][0].append(word) # begin
                    sent_group_mask[idx][0] = 1
                    for pos in range(1, word_length-1):
                        sent_words[idx+pos][1].append(word) # middle
                    sent_words[idx+word_length-1][2].append(word) # end
        if len(sent_words[idx][1]) > 0:
            sent_group_mask[idx][1] = 1
        if len(sent_words[idx][2]) > 0:
            sent_group_mask[idx][2] = 1

    return sent_words, sent_group_mask


def sent_to_matched_words(sent, lexicon_tree, max_word_num=None):
    """same to sent_to_matched_words_boundaries, but only return words"""
    sent_length = len(sent)
    sent_words = [[] for _ in range(sent_length)]

    for idx in range(sent_length):
        sub_sent = sent[idx:idx + lexicon_tree.max_depth]  # speed using max depth
        words = lexicon_tree.enumerateMatch(sub_sent)

        if len(words) == 0:
            continue
        else:
            if len(words) == 1 and len(words[0]) == 1: # single character word
                if len(sent_words[idx]) == 0:
                    sent_words[idx].extend(words)
            else:
                if max_word_num:
                    need_num = max_word_num - len(sent_words[idx])
                    words = words[:need_num]
                sent_words[idx].extend(words)
                for word in words:
                    start_pos = idx + 1
                    end_pos = idx + len(word) - 1
                    for tmp_j in range(start_pos, end_pos):
                        sent_words[tmp_j].append(word)
                    sent_words[end_pos].append(word)

    return sent_words

def sent_to_matched_words_set(sent, lexicon_tree, max_word_num=None):
    """return matched words set"""
    sent_length = len(sent)
    sent_words = [[] for _ in range(sent_length)]
    matched_words_set = set()
    for idx in range(sent_length):
        sub_sent = sent[idx:idx + lexicon_tree.max_depth]  # speed using max depth
        words = lexicon_tree.enumerateMatch(sub_sent)

        _ = [matched_words_set.add(word) for word in words]
    matched_words_set = list(matched_words_set)
    matched_words_set = sorted(matched_words_set)
    return matched_words_set


def get_corpus_matched_word_from_vocab_files(files, vocab_files, scan_nums=None):
    """
    the corpus's matched words from vocab files
    Args:
        files: input data files
        vocab_files: input vocab files
        scan_num: -1 total,
    Returns:
        total_matched_words:
        lexicon_tree:
    """
    # 1.获取词汇表
    vocabs = set()
    if scan_nums is None:
        length = len(vocab_files)
        scan_nums = [-1] * length

    for file, need_num in zip(vocab_files, scan_nums):
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_line_num = len(lines)
            if need_num >= 0:
                total_line_num = min(total_line_num, need_num)

            line_iter = trange(total_line_num)
            for idx in line_iter:
                line = lines[idx]
                line = line.strip()
                items = line.split()
                word = items[0].strip()
                vocabs.add(word)
    vocabs = list(vocabs)
    vocabs = sorted(vocabs)
    # 2.建立词典树
    lexicon_tree = Trie()
    for word in vocabs:
        lexicon_tree.insert(word)

    total_matched_words = get_corpus_matched_word_from_lexicon_tree(files, lexicon_tree)
    return total_matched_words, lexicon_tree


def get_corpus_matched_word_from_lexicon_tree(files, lexicon_tree):
    """
    数据类型统一为json格式, {'text': , 'label': }
    Args:
        files: corpus data files
        lexicon_tree: built lexicon tree

    Return:
        total_matched_words: all found matched words
    """
    total_matched_words = set()
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_line_num = len(lines)
            line_iter = trange(total_line_num)
            for idx in line_iter:
                line = lines[idx]
                line = line.strip()

                sample = json.loads(line)
                if 'text' in sample:
                    text = sample['text']
                elif 'text_a' in sample and 'text_b' in sample:
                    text_a = sample['text_a']
                    text_b = sample['text_b']
                    text = text_a + ["[SEP]"] + text_b
                sent = [ch for ch in text]
                sent_matched_words = sent_to_matched_words_set(sent, lexicon_tree)
                _ = [total_matched_words.add(word) for word in sent_matched_words]

    total_matched_words = list(total_matched_words)
    total_matched_words = sorted(total_matched_words)
    with open("matched_word.txt", "w", encoding="utf-8") as f:
        for word in total_matched_words:
            f.write("%s\n"%(word))

    return total_matched_words

def insert_seg_vocab_to_lexicon_tree(seg_vocab, word_vocab, lexicon_tree):
    """
    通过查找seg_vocab和word_vocab的重合词, 将重合词插入到lexicon_tree里面
    Args:
        seg_vocab: seg_vocab中的词文件
        word_vocab: 全量的词文件
        lexicon_tree:
    """
    seg_words = set()
    whole_words = set()
    with open(seg_vocab, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        total_line_num = len(lines)
        line_iter = trange(total_line_num)

        for idx in line_iter:
            line = lines[idx]
            line = line.strip()
            if line:
                seg_words.add(line)

    with open(word_vocab, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        total_line_num = len(lines)
        line_iter = trange(total_line_num)

        for idx in line_iter:
            line = lines[idx]
            line = line.strip()
            if line:
                whole_words.add(line)

    overleap_words = seg_words & whole_words
    overleap_words = list(overleap_words)
    overleap_words = sorted(overleap_words)
    print("Overleap words number is: \n", len(overleap_words))

    for word in overleap_words:
        lexicon_tree.insert(word)

    return lexicon_tree


def build_lexicon_tree_from_vocabs(vocab_files, scan_nums=None):
    # 1.获取词汇表
    print(vocab_files)
    vocabs = set()
    if scan_nums is None:
        length = len(vocab_files)
        scan_nums = [-1] * length

    for file, need_num in zip(vocab_files, scan_nums):
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_line_num = len(lines)
            if need_num >= 0:
                total_line_num = min(total_line_num, need_num)

            line_iter = trange(total_line_num)
            for idx in line_iter:
                line = lines[idx]
                line = line.strip()
                items = line.split()
                word = items[0].strip()
                vocabs.add(word)
    vocabs = list(vocabs)
    vocabs = sorted(vocabs)
    # 2.建立词典树
    lexicon_tree = Trie()
    for word in vocabs:
        lexicon_tree.insert(word)

    return lexicon_tree


def get_all_labels_from_corpus(files, label_file, defalut_label='O'):
    """
    Args:
        files: data files
        label_file:
    """
    labels = [defalut_label]
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    label = sample['label']
                    if isinstance(label, list):
                        for l in label:
                            if l not in labels:
                                labels.append(l)
                    else:
                        labels.append(label)

    with open(label_file, 'w', encoding='utf-8') as f:
        for label in labels:
            f.write("%s\n"%(label))
