from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from feature_env import FeatureEvaluator
from utils.logger import info


class Encoder(nn.Module):
    def __init__(self,
                 layers,#层数
                 vocab_size,#词汇表大小，就是一共能处理多少个词
                 hidden_size):#隐藏层维度
        super().__init__()
        self.layers = layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)#把每个词转换成维度是hidden_size的向量

    def infer(self, x, predict_lambda, direction='-'):
        encoder_outputs, encoder_hidden, seq_emb, predict_value = self(x)
        grads_on_outputs = torch.autograd.grad(predict_value, encoder_outputs, torch.ones_like(predict_value))[0]
                           #计算预测值相对于编码器输出的梯度
        if direction == '+':
            new_encoder_outputs = encoder_outputs + predict_lambda * grads_on_outputs
        elif direction == '-':
            new_encoder_outputs = encoder_outputs - predict_lambda * grads_on_outputs
        else:
            raise ValueError('Direction must be + or -, got {} instead'.format(direction))
        new_encoder_outputs = F.normalize(new_encoder_outputs, 2, dim=-1)
        new_seq_emb = torch.mean(new_encoder_outputs, dim=1)
        new_seq_emb = F.normalize(new_seq_emb, 2, dim=-1)
        return encoder_outputs, encoder_hidden, seq_emb, predict_value, new_encoder_outputs, new_seq_emb

    def forward(self, x):
        pass


class RNNEncoder(Encoder):
    def __init__(self,
                 layers,
                 vocab_size,
                 hidden_size,
                 dropout,
                 mlp_layers,
                 mlp_hidden_size,
                 mlp_dropout
                 ):
        super(RNNEncoder, self).__init__(layers, vocab_size, hidden_size)

        self.mlp_layers = mlp_layers
        self.mlp_hidden_size = mlp_hidden_size

        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.layers, batch_first=True, dropout=dropout)
        self.mlp = nn.Sequential()
        for i in range(self.mlp_layers):
            if i == 0:
                self.mlp.add_module('layer_{}'.format(i), nn.Sequential(
                    nn.Linear(self.hidden_size, self.mlp_hidden_size),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=mlp_dropout)))
            else:
                self.mlp.add_module('layer_{}'.format(i), nn.Sequential(
                    nn.Linear(self.mlp_hidden_size, self.mlp_hidden_size),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=mlp_dropout)))
        self.regressor = nn.Linear(self.hidden_size if self.mlp_layers == 0 else self.mlp_hidden_size, 1)

    def forward(self, x):
        embedded = self.embedding(x)  # batch x length x hidden_size
        embedded = self.dropout(embedded)

        out, hidden = self.rnn(embedded)
        out = F.normalize(out, 2, dim=-1)
        encoder_outputs = out  # final output
        encoder_hidden = hidden  # layer-wise hidden

        out = torch.mean(out, dim=1)
        out = F.normalize(out, 2, dim=-1)
        seq_emb = out

        out = self.mlp(out)
        out = self.regressor(out)
        predict_value = torch.sigmoid(out)
        return encoder_outputs, encoder_hidden, seq_emb, predict_value


def construct_encoder(fe: FeatureEvaluator, args) -> Encoder:
    name = args.method_name#在train里的设置，现在只有rnn
    size = fe.ds_size #数据集大小
    info(f'Construct Encoder with method {name}...')
    if name == 'rnn':
        return RNNEncoder(
            layers=args.encoder_layers,#train里面设置，现在只有1层
            vocab_size=size + 1, #数据集大小+1，
            hidden_size=args.encoder_hidden_size,
            dropout=args.encoder_dropout,
            mlp_layers=args.mlp_layers,#train里面设置，现在只有2层
            mlp_hidden_size=args.mlp_hidden_size,#train里面设置，现在只有200
            mlp_dropout=args.encoder_dropout
        )
    elif name == 'transformer':
        assert False
    else:
        assert False
