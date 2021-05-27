# -*- coding: utf-8 -*-
# @Time    : 2020/11/26 12:47
# @Author  : liuwei
# @File    : wcbert_parser.py
import argparse

def get_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data/dataset/NER", type=str, help="The input data dir")
    parser.add_argument("--output_dir", default="data/result", type=str, help="the output dir")
    parser.add_argument("--overwrite_cache", default=True, help="overwrite the cache or not")
    parser.add_argument("--logging_dir", default='data/log', type=str, help="the dir for log")

    ## for from_pretrained parameters
    parser.add_argument("--model_name_or_path", default=None, type=str, help="the pretrained bert path")
    parser.add_argument("--model_type", default="Bert_Token", type=str,
                        help="Bert_Token, BertCRF_Token, BertBiLSTMCRF_Token, WCBert_Token, WC....")
    parser.add_argument("--config_name", default="data/berts/bert/config.json", type=str, help="the config of define model")
    parser.add_argument("--vocab_file", default="data/berts/bert/vocab.txt", type=str, help="the vocab file for bert")
    parser.add_argument("--word_vocab_file", default="data/vocab/final_vocab.txt", type=str)
    parser.add_argument("--label_file", default="data/dataset/NER/label.txt", type=str)
    parser.add_argument("--default_label", default='O', type=str)
    parser.add_argument("--word_embedding", default="data/embedding/word_embedding.txt",
                        help="the embedding file path")
    parser.add_argument("--saved_embedding_dir", default="data/embedding", type=str)

    parser.add_argument("--do_train", default=False, action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", default=False, action="store_true", help="Whether to do evaluation")
    parser.add_argument("--do_predict", default=False, action="store_true")
    parser.add_argument("--evaluate_during_training", default=False, action="store_true",
                        help="Whether do evuation during training.")
    parser.add_argument("--max_seq_length", default=48, type=int, help="the max length of input sequence")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,  help="the training batch size")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int, help="the eval batch size")
    parser.add_argument("--num_train_epochs", default=2, type=int, help="training epoch, only work when max_step==-1")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="the weight of L2 normalization")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer")
    parser.add_argument("--sgd_momentum", default=0.9, type=float, help="momentum value for SGD")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="max clip gradient?")
    parser.add_argument("--max_steps", default=-1, type=int, help="the total number of training steps")
    parser.add_argument("--warmup_steps", default=95, type=int, help="the number of warmup steps")
    parser.add_argument("--save_steps", default=800, type=int, help="How often to save the model chekcpoint")
    parser.add_argument("--save_total_limit", default=50, type=int, help="the total number of saved checkpoints")
    parser.add_argument("--seed", default=106524, type=int, help="the seed used to initiate parameters")
    parser.add_argument("--logging_steps", default=4, type=int, help="Log every X updates steps")
    parser.add_argument("--do_shuffle", default=True, type=bool, help="do shuffle for each piece dataset or not")
    parser.add_argument("--word_embed_dim", default=200, type=int, help="the dimension of item embedding")
    parser.add_argument("--max_scan_num", default=10000, type=int, help="The boundary of data files")
    parser.add_argument("--max_word_num", default=5, type=int)

    ## machine parameter
    parser.add_argument("--no_cuda", default=False, help="Do not use CUDA even it is available")
    parser.add_argument("--fp16", default=False, action="store_true", help="Whether use fp16 to old_train")
    parser.add_argument("--fp16_opt_level", default="O1", type=str,
                        help="level selected in ['O0', 'O1', 'O2', 'O3']")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulatate before performing update")

    # for distribute training
    parser.add_argument("--nodes", default=1, type=int,
                        help="the total number of nodes(machines) we are going to use")
    parser.add_argument("--n_gpu", default=1, type=int,
                        help="ranking within the nodes")
    parser.add_argument("--local_rank", default=-1, type=int,
                        help="the rank of current node within all nodes, goes from 0 to args.nodes-1")


    return parser