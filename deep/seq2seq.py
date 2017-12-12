# -*- coding:utf-8 -*-
"""
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
from chainer import cuda

# global variable (initialize)
xp = np


class Encoder(chainer.Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, batch_size):
        super(Encoder, self).__init__(
            xe=L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            enc_lstm1=L.LSTM(embed_size, hidden_size),
            enc_lstm2=L.LSTM(hidden_size, hidden_size),
        )
        self.hidden_size = hidden_size
        self.batch_size = batch_size

    def reset_state(self):
        self.enc_lstm1.reset_state()
        self.enc_lstm2.reset_state()

    def __call__(self, x, train=True):
        e = F.tanh(self.xe(x))
        enc_h1 = self.enc_lstm1(e)
        enc_h2 = self.enc_lstm2(enc_h1)


class Decoder(chainer.Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Decoder, self).__init__(
            ye=L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            dec_lstm1=L.LSTM(embed_size, hidden_size),
            dec_lstm2=L.LSTM(hidden_size, hidden_size),
            dec_out=L.Linear(hidden_size, vocab_size),
        )

    def reset_state(self):
        self.dec_lstm1.reset_state()
        self.dec_lstm2.reset_state()

    def __call__(self, y):
        e = F.tanh(self.ye(y))
        dec_h1 = self.dec_lstm1(e)
        dec_h2 = self.dec_lstm2(dec_h1)
        return self.dec_out(dec_h2)


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

        super(Seq2Seq, self).__init__(
            # encoder
            enc=Encoder(vocab_size, feature_num, hidden_num, batch_size),

            # decoder initializer
            enc_dec_1_c=L.Linear(hidden_num, hidden_num),
            enc_dec_1_h=L.Linear(hidden_num, hidden_num),
            enc_dec_2_c=L.Linear(hidden_num, hidden_num),
            enc_dec_2_h=L.Linear(hidden_num, hidden_num),

            # decoder
            dec=Decoder(vocab_size, feature_num, hidden_num)
        )

    def encode(self, input_batch, train):
        """
        Input batch of sequence and update self.c (context vector) and self.h (hidden vector)
        :param input_batch: batch of input text embed id ex.) [[ 1, 0 ,14 ,5 ], [ ...] , ...]
        :param train : True or False
        """
        for batch_word in input_batch:
            batch_word = chainer.Variable(xp.array(batch_word, dtype=xp.int32))
            self.enc(batch_word, train=train)

    def init_decoder(self):
        dec_c1 = self.enc_dec_1_c(self.enc.enc_lstm1.c)
        dec_h1 = self.enc_dec_1_h(self.enc.enc_lstm1.h)
        dec_c2 = self.enc_dec_2_c(self.enc.enc_lstm2.c)
        dec_h2 = self.enc_dec_1_h(self.enc.enc_lstm2.h)
        self.dec.dec_lstm1.set_state(dec_c1, dec_h1)
        self.dec.dec_lstm2.set_state(dec_c2, dec_h2)

    def decode(self, predict_id, teacher_id, train):
        """
        :param predict_id: batch of word ID by output of decoder
        :param teacher_id : batch of correct ID
        :param train: True or false
        :return: decoded embed vector
        """
        batch_word = chainer.Variable(xp.array(predict_id, dtype=xp.int32))
        predict_mat = self.dec(batch_word)
        if train:
            t = xp.array(teacher_id, dtype=xp.int32)
            t = chainer.Variable(t)
            return F.softmax_cross_entropy(predict_mat, t), predict_mat
        else:
            return predict_mat

    def initialize(self):
        self.enc.reset_state()
        self.dec.reset_state()

    def one_encode(self, src_text, train):
        """
        :param src_text: input text embed id ex.) [ 1, 0 ,14 ,5 ]
        :param train : True or False
        :return: context vector (hidden vector)
        """
        for word in src_text:
            word = chainer.Variable(xp.array([word], dtype=xp.int32))
            self.enc(word, train=train)

    def one_decode(self, predict_id, teacher_id, train):
        """
        :param predict_id:
        :param teacher_id : embed id ( teacher's )
        :param train: True or false
        :return: decoded embed vector
        """
        word = chainer.Variable(xp.array([predict_id], dtype=xp.int32))
        predict_vec = self.dec(word)
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
        self.init_decoder()

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

