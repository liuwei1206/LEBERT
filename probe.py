"""
We use this code to probe if the compress is correct
"""

import os
import json
import random
import time
import torch
from tqdm import tqdm, trange
import numpy as np
from transformers import BertTokenizer
from function.preprocess import build_lexicon_tree_from_vocabs, get_corpus_matched_word_from_lexicon_tree
from feature.vocab import ItemVocabArray, ItemVocabFile


#################### 1.probe words ###################
"""
# 1. prepare vocabs
tokenizer = BertTokenizer.from_pretrained("data/berts/bert/vocab.txt")

task_name = "data/dataset/NER/weibo"
lexicon_tree = build_lexicon_tree_from_vocabs(["data/vocab/tencent_vocab.txt"], scan_nums=[1500000])
embed_lexicon_tree = lexicon_tree
train_data_file = os.path.join(task_name, "train.json")
dev_data_file = os.path.join(task_name, "dev.json")
test_data_file = os.path.join(task_name, "test.json")
data_files = [train_data_file, dev_data_file, test_data_file]
matched_words = get_corpus_matched_word_from_lexicon_tree(data_files, embed_lexicon_tree)
word_vocab = ItemVocabArray(items_array=matched_words, is_word=True, has_default=False, unk_num=5)
label_vocab = ItemVocabFile(files=[os.path.join(task_name, "labels.txt")], is_word=False)

# 2.load data
np_file = "data/dataset/NER/weibo/saved_maxword_5_maxseq_256_test_1500000.npz"
with np.load(np_file) as dataset:
    input_ids = dataset["input_ids"]
    attention_mask = dataset["attention_mask"]
    input_matched_word_ids = dataset["input_matched_word_ids"]
    labels = dataset["labels"]


# 3. we print some examples
for i in range(2):
    sent_length = np.sum(attention_mask[i])
    print(tokenizer.convert_ids_to_tokens(input_ids[i][:sent_length]))
    for j in range(sent_length):
        print(word_vocab.convert_ids_to_items(input_matched_word_ids[i][j]))
    print(label_vocab.convert_ids_to_items(labels[i][:sent_length]))

"""
########################### 2. probe embedding ###############33
# 10343 团结
# 10346 团结的力量
task_name = "data/dataset/NER/weibo"
saved_corpus_embedding_file = os.path.join(task_name, "saved_word_embedding_1500000.pkl")
embedding_path = "data/embedding/word_embedding.txt"
max_scan_num = 1500000

with open(saved_corpus_embedding_file, 'rb') as f:
    pretrained_emb = pickle.load(f)
    print(pretrained_emb[10343])

with open(embedding_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    max_scan_num = min(max_scan_num, len(lines))
    line_iter = trange(max_scan_num)
    for idx in line_iter:
        line = lines[idx]
        line = line.strip()
        items = line.split()
        if items[0].strip() == "团结":
            print(items[1:])
            break

