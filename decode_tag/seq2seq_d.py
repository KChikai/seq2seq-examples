# -*- coding:utf-8 -*-
"""
Add Simple Seq2Seq to Speaker Model (Decoder Tag).
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
    def __init__(self, vocab_size, embed_size, hidden_size, batch_size):
        super(Encoder, self).__init__(
            xe=L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            eh=L.Linear(embed_size, 4 * hidden_size),
            hh=L.Linear(hidden_size, 4 * hidden_size),
        )
        self.hidden_size = hidden_size
        self.batch_size = batch_size

    def __call__(self, x, c_pre, h_pre, train=True):
        e = F.tanh(self.xe(x))
        c_tmp, h_tmp = F.lstm(c_pre, self.eh(e) + self.hh(h_pre))
        enable = chainer.Variable(chainer.Variable(x.data != -1).data.reshape(len(x), 1))    # calculate flg whether x is -1 or not
        c_next = F.where(enable, c_tmp, c_pre)                                   # if x!=-1, c_tmp . elseif x=-1, c_pre.
        h_next = F.where(enable, h_tmp, h_pre)                                   # if x!=-1, h_tmp . elseif x=-1, h_pre.
        return c_next, h_next


# speaker model's decoder
class Decoder(chainer.Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, label_size, label_embed_size):
        super(Decoder, self).__init__(
            ye=L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            le=L.EmbedID(label_size, label_embed_size, ignore_label=-1),
            eh=L.Linear(embed_size + label_embed_size, 4 * hidden_size),
            hh=L.Linear(hidden_size, 4 * hidden_size),
            hf=L.Linear(hidden_size, embed_size),
            fy=L.Linear(embed_size, vocab_size),
        )

    def __call__(self, y, y_label, c_pre, h_pre):
        e = F.tanh(self.ye(y))
        e_l = F.tanh(self.le(y_label))
        c_tmp, h_tmp = F.lstm(c_pre, self.eh(F.concat((e, e_l))) + self.hh(h_pre))
        enable = chainer.Variable(chainer.Variable(y.data != -1).data.reshape(len(y), 1))
        c_next = F.where(enable, c_tmp, c_pre)
        h_next = F.where(enable, h_tmp, h_pre)
        f = F.tanh(self.hf(h_next))
        return self.fy(f), c_next, h_next


class Seq2Seq(chainer.Chain):

    def __init__(self, vocab_size, feature_num, hidden_num, label_num,
                 label_embed_num, batch_size, gpu_flg):
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
        self.c_batch = Variable(xp.zeros((batch_size, self.hidden_num), dtype=xp.float32))  # cell Variable
        self.h_batch = Variable(xp.zeros((batch_size, self.hidden_num), dtype=xp.float32))  # hidden Variable

        super(Seq2Seq, self).__init__(
            enc=Encoder(vocab_size, feature_num, hidden_num, batch_size),
            dec=Decoder(vocab_size, feature_num, hidden_num, label_num, label_embed_num)
        )

    def encode(self, input_batch, train):
        """
        Input batch of sequence and update self.c (context vector) and self.h (hidden vector)
        :param input_batch: batch of input text embed id ex.) [[ 1, 0 ,14 ,5 ], [ ...] , ...]
        :param train : True or False
        """
        for batch_word in input_batch:
            batch_word = chainer.Variable(xp.array(batch_word, dtype=xp.int32))
            self.c_batch, self.h_batch = self.enc(batch_word, self.c_batch, self.h_batch, train=train)

    def decode(self, predict_id, teacher_id, label_id, train):
        """
        :param predict_id: batch of word ID by output of decoder
        :param teacher_id : batch of correct ID
        :param label_id : batch of label ID
        :param train: True or false
        :return: decoded embed vector
        """
        batch_word = chainer.Variable(xp.array(predict_id, dtype=xp.int32))
        batch_label = chainer.Variable(xp.array(label_id, dtype=xp.int32))
        predict_mat, self.c_batch, self.h_batch = self.dec(batch_word, batch_label, self.c_batch, self.h_batch)
        if train:
            t = xp.array(teacher_id, dtype=xp.int32)
            t = chainer.Variable(t)
            return F.softmax_cross_entropy(predict_mat, t), predict_mat
        else:
            return predict_mat

    def initialize(self):
        self.c_batch = Variable(xp.zeros((self.batch_size, self.hidden_num), dtype=xp.float32))
        self.h_batch = Variable(xp.zeros((self.batch_size, self.hidden_num), dtype=xp.float32))

    def one_encode(self, src_text, train):
        """
        :param src_text: input text embed id ex.) [ 1, 0 ,14 ,5 ]
        :param train : True or False
        :return: context vector (hidden vector)
        """
        for word in src_text:
            word = chainer.Variable(xp.array([word], dtype=xp.int32))
            self.c_batch, self.h_batch = self.enc(word, self.c_batch, self.h_batch, train=train)

    def one_decode(self, predict_id, teacher_id, label_id, train):
        """
        :param predict_id:
        :param teacher_id : embed id ( teacher's )
        :param label_id : label id
        :param train: True or false
        :return: decoded embed vector
        """
        word = chainer.Variable(xp.array([predict_id], dtype=xp.int32))
        label = chainer.Variable(xp.array([label_id], dtype=xp.int32))
        predict_vec, self.c_batch, self.h_batch = self.dec(word, label, self.c_batch, self.h_batch)
        if train:
            t = xp.array([teacher_id], dtype=xp.int32)
            t = chainer.Variable(t)
            return F.softmax_cross_entropy(predict_vec, t), predict_vec
        else:
            return predict_vec

    def generate(self, src_text, sentence_limit, word2id, id2word, label_id):
        """
        :param src_text: input text embed id ex.) [ 1, 0 ,14 ,5 ]
        :param sentence_limit:
        :param word2id:
        :param id2word:
        :param label_id: label id
        :return:
        """
        self.initialize()
        self.one_encode(src_text, train=False)

        sentence = ""
        word_id = word2id["<eos>"]          # <start>タグを使わないバージョン
        for _ in range(sentence_limit):
            predict_vec = self.one_decode(predict_id=word_id, teacher_id=None, label_id=label_id, train=False)
            word = id2word[xp.argmax(predict_vec.data)]     # choose word_ID which has the highest probability
            word_id = word2id[word]
            if word == "<eos>":
                break
            sentence = sentence + word + " "
        return sentence

