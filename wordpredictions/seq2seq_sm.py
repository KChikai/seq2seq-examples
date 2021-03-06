# -*- coding:utf-8 -*-
"""
Word Prediction Model (using Softmax)
あとAdamからAdaDeltaに変更した．
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
        enable = chainer.Variable(chainer.Variable(x.data != -1).data.reshape(len(x), 1))
        c_next = F.where(enable, c_tmp, c_pre)
        h_next = F.where(enable, h_tmp, h_pre)
        return c_next, h_next


class WPe(chainer.Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(WPe, self).__init__(
            ws=L.Linear(hidden_size, hidden_size),
            wattp=L.Linear(hidden_size*2, hidden_size),
            wt=L.Linear(hidden_size*2, embed_size),
            wf=L.Linear(embed_size, vocab_size),
        )

    def __call__(self, hs_enc):
        h_average = 0
        for h in hs_enc:
            h_average += h
        h_average /= len(hs_enc)
        s0 = F.sigmoid(self.ws(h_average))
        cp = self.caluculate_cp(s0, hs_enc)
        t = F.tanh(self.wt(F.concat((s0, cp))))
        return self.wf(t), s0

    def caluculate_cp(self, s0, hs_enc):
        sum_value = 0
        products = []
        for h_enc in hs_enc:
            result_vec = F.tanh(self.wattp(F.concat((s0, h_enc))))
            products.append(result_vec)
            sum_value += result_vec
        cp = 0
        for i, h_enc in enumerate(hs_enc):
            alpha_i = products[i] / sum_value
            cp += alpha_i * h_enc
        return cp


class Decoder(chainer.Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Decoder, self).__init__(
            ye=L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            eh=L.Linear(embed_size, 4 * hidden_size),
            hh=L.Linear(hidden_size, 4 * hidden_size),
            wc=L.Linear(hidden_size, hidden_size),
            wh=L.Linear(hidden_size, hidden_size),
            fy=L.Linear(hidden_size, vocab_size),
        )

    def __call__(self, y, c_pre, h_pre, hs_enc):
        e = F.tanh(self.ye(y))
        c_tmp, h_tmp = F.lstm(c_pre, self.eh(e) + self.hh(h_pre))
        enable = chainer.Variable(chainer.Variable(y.data != -1).data.reshape(len(y), 1))
        c_next = F.where(enable, c_tmp, c_pre)
        h_next = F.where(enable, h_tmp, h_pre)
        ct = self.calculate_alpha(h_next, hs_enc)
        f = F.tanh(self.wc(ct) + self.wh(h_next))
        return self.fy(f), c_next, h_next

    @staticmethod
    def calculate_alpha(h, hs_enc):
        sum_value = Variable(xp.zeros((h.shape[0], 1), dtype=xp.float32))
        products = []
        for h_enc in hs_enc:
            inner_product = F.exp(F.batch_matmul(h, h_enc, transa=True) * (1 / 10e+03)).data[:, :, 0]
            products.append(inner_product)
            sum_value += inner_product
        ct = Variable(xp.zeros((h.shape[0], h.shape[1]), dtype=xp.float32))
        for i, h_enc in enumerate(hs_enc):
            alpha_i = (products[i]  * (1 / 10e+03)) / sum_value
            ct += alpha_i.data * h_enc.data
        return ct


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
        self.c_batch = Variable(xp.zeros((batch_size, self.hidden_num), dtype=xp.float32))  # cell Variable
        self.h_batch = Variable(xp.zeros((batch_size, self.hidden_num), dtype=xp.float32))  # hidden Variable
        self.h_enc = []                                                                     # for calculating alpha

        super(Seq2Seq, self).__init__(
            enc=Encoder(vocab_size, feature_num, hidden_num, batch_size),       # encoder
            wpe=WPe(vocab_size, feature_num, hidden_num),
            dec=Decoder(vocab_size, feature_num, hidden_num)                    # decoder
        )

    def encode(self, input_batch, teacher_wp, train):
        """
        Input batch of sequence and update self.c (context vector) and self.h (hidden vector)
        :param teacher_wp:
        :param input_batch: batch of input text embed id ex.) [[ 1, 0 ,14 ,5 ], [ ...] , ...]
        :param train : True or False
        """
        for batch_word in input_batch:
            batch_word = chainer.Variable(xp.array(batch_word, dtype=xp.int32))
            self.c_batch, self.h_batch = self.enc(batch_word, self.c_batch, self.h_batch, train=train)
            self.h_enc.append(self.h_batch)
        predict_mat, self.h_batch = self.wpe(self.h_enc)

        if train:
            return -F.sum(F.log(F.softmax(predict_mat)) * teacher_wp)

    def decode(self, predict_id, teacher_id, train):
        """
        :param predict_id: batch of word ID by output of decoder
        :param teacher_id : batch of correct ID
        :param train: True or false
        :return: decoded embed vector
        """
        batch_word = chainer.Variable(xp.array(predict_id, dtype=xp.int32))
        predict_mat, self.c_batch, self.h_batch = self.dec(batch_word, self.c_batch, self.h_batch, self.h_enc)
        if train:
            t = xp.array(teacher_id, dtype=xp.int32)
            t = chainer.Variable(t)
            return F.softmax_cross_entropy(predict_mat, t), predict_mat
        else:
            return predict_mat

    def initialize(self):
        self.c_batch = Variable(xp.zeros((self.batch_size, self.hidden_num), dtype=xp.float32))
        self.h_batch = Variable(xp.zeros((self.batch_size, self.hidden_num), dtype=xp.float32))
        self.h_enc.clear()

    def one_encode(self, src_text, train):
        """
        とりあえずはテスト時は Word Prediction の出力はしない
        :param src_text: input text embed id ex.) [ 1, 0 ,14 ,5 ]
        :param train : True or False
        :return: context vector (hidden vector)
        """
        for word in src_text:
            word = chainer.Variable(xp.array([word], dtype=xp.int32))
            self.c_batch, self.h_batch = self.enc(word, self.c_batch, self.h_batch, train=train)
            self.h_enc.append(self.h_batch)
        predict_mat, self.h_batch = self.wpe(self.h_enc)

        # if train:
        #     predict_mat = self.wpe(self.h_enc)
        #     return F.mean_squared_error(predict_mat, teacher_wp)

    def one_decode(self, predict_id, teacher_id, train):
        """
        :param predict_id:
        :param teacher_id : embed id ( teacher's )
        :param train: True or false
        :return: decoded embed vector
        """
        word = chainer.Variable(xp.array([predict_id], dtype=xp.int32))
        predict_vec, self.c_batch, self.h_batch = self.dec(word, self.c_batch, self.h_batch, self.h_enc)
        if train:
            t = xp.array([teacher_id], dtype=xp.int32)
            t = chainer.Variable(t)
            return F.softmax_cross_entropy(predict_vec, t), predict_vec
        else:
            return predict_vec

    def generate(self, src_text, sentence_limit, word2id, id2word):
        """
        :param src_text: input text embed id ex.) [ 1, 0 ,14 ,5 ]
        :param sentence_limit:
        :param word2id:
        :param id2word:
        :return:
        """
        self.initialize()
        self.one_encode(src_text, train=False)

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

