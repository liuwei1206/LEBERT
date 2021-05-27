# -*- coding: utf-8 -*-
# @Time    : 2020/11/30 12:47
# @Author  : liuwei
# @File    : bilstm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from function.utils import reverse_padded_sequence

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(BiLSTM, self).__init__()

        self.f_lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.b_lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, attention_mask):
        batch_length = torch.sum(attention_mask, dim=-1) # [batch]
        batch_length = list(map(int, batch_length))

        f_lstm_output, _ = self.f_lstm(inputs)
        b_lstm_output, _ = self.b_lstm(inputs)

        f_lstm_output = self.dropout(f_lstm_output)
        b_lstm_output = self.dropout(b_lstm_output)

        # reverse
        b_lstm_output = reverse_padded_sequence(b_lstm_output, batch_length)

        # concat
        lstm_output = torch.cat((f_lstm_output, b_lstm_output), dim=-1)

        return lstm_output


