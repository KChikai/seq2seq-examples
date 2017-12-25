# -*- coding:utf-8 -*-
"""
双方向seq2seqモデル．
アテンションがついていないモデル．

Sample script of Sequence to Sequence model.
You can also use Batch and GPU.
This model is based on below paper.

Ilya Sutskever, Oriol Vinyals, and Quoc V. Le.
Sequence to sequence learning with neural networks.
In Advances in Neural Information Processing Systems (NIPS 2014).
"""
import os
os.environ["CHAINER_TYPE_CHECK"] = "0"

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable

# global variable (initialize)
xp = np


class Encoder(chainer.Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Encoder, self).__init__(
            # embed
            xe=L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            # forward lstm
            eh=L.Linear(embed_size, 4 * hidden_size),
            hh=L.Linear(hidden_size, 4 * hidden_size),
            # backward lstm
            eh_rev=L.Linear(embed_size, 4 * hidden_size),
            hh_rev=L.Linear(hidden_size, 4 * hidden_size),
        )

    def __call__(self, x, c_pre, h_pre, x_rev, c_pre_rev, h_pre_rev, train=True):
        # forward lstm
        e = F.tanh(self.xe(x))
        c_tmp, h_tmp = F.lstm(c_pre, F.dropout(self.eh(e), ratio=0.2, train=train) + self.hh(h_pre))
        enable = chainer.Variable(chainer.Variable(x.data != -1).data.reshape(len(x), 1))   # calculate flg whether x is -1 or not
        c_next = F.where(enable, c_tmp, c_pre)                                              # if x!=-1, c_tmp . elseif x=-1, c_pre.
        h_next = F.where(enable, h_tmp, h_pre)                                              # if x!=-1, h_tmp . elseif x=-1, h_pre.

        # backward lstm
        e_rev = F.tanh(self.xe(x_rev))
        c_tmp_rev, h_tmp_rev = F.lstm(c_pre_rev, F.dropout(self.eh_rev(e_rev), ratio=0.2, train=train) + self.hh_rev(h_pre_rev))
        enable_rev = chainer.Variable(chainer.Variable(x_rev.data != -1).data.reshape(len(x), 1))   # calculate flg whether x is -1 or not
        c_next_rev = F.where(enable_rev, c_tmp_rev, c_pre_rev)                                      # if x!=-1, c_tmp . elseif x=-1, c_pre.
        h_next_rev = F.where(enable_rev, h_tmp_rev, h_pre_rev)                                      # if x!=-1, h_tmp . elseif x=-1, h_pre.

        return c_next, h_next, c_next_rev, h_next_rev


class Decoder(chainer.Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Decoder, self).__init__(
            ye=L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            eh=L.Linear(embed_size, 4 * hidden_size),
            hh=L.Linear(hidden_size, 4 * hidden_size),
            hf=L.Linear(hidden_size, embed_size),
            fy=L.Linear(embed_size, vocab_size),
        )

    def __call__(self, y, c_pre, h_pre, train=True):
        e = F.tanh(self.ye(y))
        c_tmp, h_tmp = F.lstm(c_pre, F.dropout(self.eh(e), ratio=0.2, train=train) + self.hh(h_pre))
        enable = chainer.Variable(chainer.Variable(y.data != -1).data.reshape(len(y), 1))
        c_next = F.where(enable, c_tmp, c_pre)
        h_next = F.where(enable, h_tmp, h_pre)
        f = F.tanh(self.hf(h_next))
        return self.fy(f), c_next, h_next


class Seq2Seq(chainer.Chain):

    def __init__(self, vocab_size, feature_num, hidden_num, batch_size, gpu_flg):
        """
        :param vocab_size: input vocab size
        :param feature_num: size of feature layer (embed layer)
        :param hidden_num: size of hidden layer
        :return:
        """
        global xp
        xp = cuda.cupy if gpu_flg >= 0 else np

        self.vocab_size = vocab_size
        self.hidden_num = hidden_num
        self.batch_size = batch_size
        self.c_batch = Variable(xp.zeros((batch_size, self.hidden_num), dtype=xp.float32))
        self.h_batch = Variable(xp.zeros((batch_size, self.hidden_num), dtype=xp.float32))
        self.c_batch_rev = Variable(xp.zeros((batch_size, self.hidden_num), dtype=xp.float32))
        self.h_batch_rev = Variable(xp.zeros((batch_size, self.hidden_num), dtype=xp.float32))
        self.h_enc = []                                                         # for calculating alpha

        super(Seq2Seq, self).__init__(
            enc=Encoder(vocab_size, feature_num, hidden_num),                   # encoder
            ws=L.Linear(hidden_num * 2, hidden_num),                            # TODO: エンコーダの隠れ層をデコーダに渡す
            dec=Decoder(vocab_size, feature_num, hidden_num)                    # decoder
        )

    def encode(self, input_batch, input_batch_rev, train):
        """
        Input batch of sequence and update self.c (context vector) and self.h (hidden vector)
        :param input_batch_rev:
        :param input_batch: batch of input text embed id ex.) [[ 1, 0 ,14 ,5 ], [ ...] , ...]
        :param train : True or False
        """
        # check the size of batch lists
        if len(input_batch) != len(input_batch_rev):
            print('Input batch must be the same size to input batch reversed.')
            raise ValueError

        # encoding
        for batch_word, batch_word_rev in zip(input_batch, input_batch_rev):
            batch_word = chainer.Variable(xp.array(batch_word, dtype=xp.int32))
            batch_word_rev = chainer.Variable(xp.array(batch_word_rev, dtype=xp.int32))
            self.c_batch, self.h_batch, self.c_batch_rev, self.h_batch_rev = self.enc(batch_word, self.c_batch, self.h_batch,
                                                                                      batch_word_rev, self.c_batch_rev, self.h_batch_rev, train=train)
            self.h_enc.append(F.concat((self.h_batch, self.h_batch_rev)))

        # calculate initial state
        h_average = 0
        for h in self.h_enc:
            h_average += h
        h_average /= len(self.h_enc)
        self.h_batch = F.sigmoid(self.ws(h_average))
        # self.c_batch = F.concat((self.c_batch, self.c_batch_rev))
        self.c_batch = self.c_batch                                 # TODO: とりあえずencoderの片方のcell batchを渡している

    def decode(self, predict_id, teacher_id, train):
        """
        :param predict_id: batch of word ID by output of decoder
        :param teacher_id : batch of correct ID
        :param train: True or false
        :return: decoded embed vector
        """
        batch_word = chainer.Variable(xp.array(predict_id, dtype=xp.int32))
        predict_mat, self.c_batch, self.h_batch = self.dec(batch_word, self.c_batch, self.h_batch, train=train)
        if train:
            t = xp.array(teacher_id, dtype=xp.int32)
            t = chainer.Variable(t)
            return F.softmax_cross_entropy(predict_mat, t), predict_mat
        else:
            return predict_mat

    def initialize(self, batch_size):
        self.c_batch = Variable(xp.zeros((batch_size, self.hidden_num), dtype=xp.float32))
        self.h_batch = Variable(xp.zeros((batch_size, self.hidden_num), dtype=xp.float32))
        self.c_batch_rev = Variable(xp.zeros((batch_size, self.hidden_num), dtype=xp.float32))
        self.h_batch_rev = Variable(xp.zeros((batch_size, self.hidden_num), dtype=xp.float32))
        self.h_enc.clear()

    def one_encode(self, src_text, src_text_rev, train):
        """
        :param src_text: input text embed id ex.) [ 1, 0 ,14 ,5 ]
        :param src_text_rev:
        :param train : True or False
        :return: context vector (hidden vector)
        """
        for word, word_rev in zip(src_text, src_text_rev):
            word = chainer.Variable(xp.array([word], dtype=xp.int32))
            word_rev = chainer.Variable(xp.array([word_rev], dtype=xp.int32))
            self.c_batch, self.h_batch, self.c_batch_rev, self.h_batch_rev = self.enc(word, self.c_batch, self.h_batch,
                                                                                      word_rev, self.c_batch_rev, self.h_batch_rev, train=train)
            self.h_enc.append(F.concat((self.h_batch, self.h_batch_rev)))

        # calculate initial state
        h_average = 0
        for h in self.h_enc:
            h_average += h
        h_average /= len(self.h_enc)
        self.h_batch = F.sigmoid(self.ws(h_average))
        # self.c_batch = F.concat((self.c_batch, self.c_batch_rev))
        self.c_batch = self.c_batch                                 # TODO: とりあえずencoderの片方のcell batchを渡している

    def one_decode(self, predict_id, teacher_id, train):
        """
        :param predict_id:
        :param teacher_id : embed id ( teacher's )
        :param train: True or false
        :return: decoded embed vector
        """
        word = chainer.Variable(xp.array([predict_id], dtype=xp.int32))
        predict_vec, self.c_batch, self.h_batch = self.dec(word, self.c_batch, self.h_batch, train=train)
        if train:
            t = xp.array([teacher_id], dtype=xp.int32)
            t = chainer.Variable(t)
            return F.softmax_cross_entropy(predict_vec, t), predict_vec
        else:
            return predict_vec

    def generate(self, src_text, src_text_rev, sentence_limit, word2id, id2word):
        """
        :param src_text: input text embed id ex.) [ 1, 0 ,14 ,5 ]
        :param src_text_rev:
        :param sentence_limit:
        :param word2id:
        :param id2word:
        :return:
        """
        self.initialize(batch_size=1)
        self.one_encode(src_text, src_text_rev, train=False)

        sentence = ""
        word_id = word2id["<start>"]
        for _ in range(sentence_limit):
            predict_vec = self.one_decode(predict_id=word_id, teacher_id=None, train=False)
            word = id2word[xp.argmax(predict_vec.data)]     # choose word_ID which has the highest probability
            word_id = word2id[word]
            if word == "<eos>":
                break
            sentence = sentence + word + " "
        return sentence

