# -*- coding: utf-8 -*-
# @Time    : 2020/11/26 15:47
# @Author  : liuwei
# @File    : vocab.py

"""
item2id vocab

for word

for labels

for boundary

"""

import os
import json

class ItemVocabFile():
    """
    Build vocab from file.
    Note, each line is a item in vocab, or each items[0] is in vocab
    """
    def __init__(self, files, is_word=False, has_default=False, unk_num=0):
        self.files = files
        self.item2idx = {}
        self.idx2item = []
        self.item_size = 0
        self.is_word = is_word
        if not has_default and self.is_word:
            self.item2idx['<pad>'] = self.item_size
            self.idx2item.append('<pad>')
            self.item_size += 1
            self.item2idx['<unk>'] = self.item_size
            self.idx2item.append('<unk>')
            self.item_size += 1
            # for unk words
            for i in range(unk_num):
                self.item2idx['<unk>{}'.format(i+1)] = self.item_size
                self.idx2item.append('<unk>{}'.format(i+1))
                self.item_size += 1

        self.init_vocab()

    def init_vocab(self):
        for file in self.files:
            with open(file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    items = line.split()
                    item = items[0].strip()
                    self.item2idx[item] = self.item_size
                    self.idx2item.append(item)
                    self.item_size += 1

    def get_item_size(self):
        return self.item_size

    def convert_item_to_id(self, item):
        if item in self.item2idx:
            return self.item2idx[item]
        elif self.is_word:
            unk = "<unk>" + str(len(item))
            if unk in self.item2idx:
                return self.item2idx[unk]
            else:
                return self.item2idx['<unk>']
        else:
            print("Label does not exist!!!!")
            print(item)
            raise KeyError()

    def convert_items_to_ids(self, items):
        return [self.convert_item_to_id(item) for item in items]

    def convert_id_to_item(self, id):
        return self.idx2item[id]

    def convert_ids_to_items(self, ids):
        return [self.convert_id_to_item(id) for id in ids]



class ItemVocabArray():
    """
    Build vocab from file.
    Note, each line is a item in vocab, or each items[0] is in vocab
    """
    def __init__(self, items_array, is_word=False, has_default=False, unk_num=0):
        self.items_array = items_array
        self.item2idx = {}
        self.idx2item = []
        self.item_size = 0
        self.is_word = is_word
        if not has_default and self.is_word:
            self.item2idx['<pad>'] = self.item_size
            self.idx2item.append('<pad>')
            self.item_size += 1
            self.item2idx['<unk>'] = self.item_size
            self.idx2item.append('<unk>')
            self.item_size += 1
            # for unk words
            for i in range(1, unk_num+1):
                self.item2idx['<unk>{}'.format(i+1)] = self.item_size
                self.idx2item.append('<unk>{}'.format(i+1))
                self.item_size += 1

        self.init_vocab()

    def init_vocab(self):
        for item in self.items_array:
            self.item2idx[item] = self.item_size
            self.idx2item.append(item)
            self.item_size += 1

    def get_item_size(self):
        return self.item_size

    def convert_item_to_id(self, item):
        if item in self.item2idx:
            return self.item2idx[item]
        elif self.is_word:
            unk = "<unk>" + str(len(item))
            if unk in self.item2idx:
                return self.item2idx[unk]
            else:
                return self.item2idx['<unk>']
        else:
            print("Label does not exist!!!!")
            print(item)
            raise KeyError()

    def convert_items_to_ids(self, items):
        return [self.convert_item_to_id(item) for item in items]

    def convert_id_to_item(self, id):
        return self.idx2item[id]

    def convert_ids_to_items(self, ids):
        return [self.convert_id_to_item(id) for id in ids]
