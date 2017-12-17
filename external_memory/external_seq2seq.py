# -*- coding:utf-8 -*-
"""
External Memory を使用したseq2seqバージョン
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
    def __init__(self, all_vocab_size, embed_size, hidden_size, batch_size):
        super(Encoder, self).__init__(
            xe=L.EmbedID(all_vocab_size, embed_size, ignore_label=-1),
            eh=L.Linear(embed_size, 4 * hidden_size),
            hh=L.Linear(hidden_size, 4 * hidden_size),
        )
        self.hidden_size = hidden_size
        self.batch_size = batch_size

    def __call__(self, x, c_pre, h_pre, train=True):
        e = F.tanh(self.xe(x))
        c_tmp, h_tmp = F.lstm(c_pre, self.eh(F.dropout(e, ratio=0.2, train=train)) + self.hh(h_pre))
        enable = chainer.Variable(chainer.Variable(x.data != -1).data.reshape(len(x), 1))    # calculate flg whether x is -1 or not
        c_next = F.where(enable, c_tmp, c_pre)                                   # if x!=-1, c_tmp . elseif x=-1, c_pre.
        h_next = F.where(enable, h_tmp, h_pre)                                   # if x!=-1, h_tmp . elseif x=-1, h_pre.
        return c_next, h_next


class Decoder(chainer.Chain):
    def __init__(self, all_vocab_size, emotion_vocab_size, embed_size, hidden_size):
        super(Decoder, self).__init__(
            ye=L.EmbedID(all_vocab_size, embed_size, ignore_label=-1),
            eh=L.Linear(embed_size, 4 * hidden_size),
            hh=L.Linear(hidden_size, 4 * hidden_size),
            vt=L.Linear(hidden_size, 1),                         # softmaxに重み付けを行ったバージョン
            wg=L.Linear(hidden_size, all_vocab_size - emotion_vocab_size),
            we=L.Linear(hidden_size, emotion_vocab_size),
        )

    def __call__(self, y, c_pre, h_pre, train=True):
        # input word embedding
        e = F.tanh(self.ye(y))

        # LSTM
        c_tmp, h_tmp = F.lstm(c_pre, self.eh(F.dropout(e, ratio=0.2, train=train)) + self.hh(h_pre))
        enable = chainer.Variable(chainer.Variable(y.data != -1).data.reshape(len(y), 1))
        c_next = F.where(enable, c_tmp, c_pre)
        h_next = F.where(enable, h_tmp, h_pre)

        # output using at
        at = F.sigmoid(self.vt(h_next))
        pg_pre = self.wg(h_next)
        pg = pg_pre * F.broadcast_to((1 - at), shape=(pg_pre.data.shape[0], pg_pre.data.shape[1]))
        pe_pre = self.we(h_next)
        pe = pe_pre * F.broadcast_to(at, shape=(pe_pre.data.shape[0], pe_pre.data.shape[1]))

        # broadcast を使わない ver.
        # pg = chainer.Variable(self.wg(h_next).data * (1 - at).data)
        # pe = chainer.Variable(self.we(h_next).data * at.data)
        return F.concat((pg, pe)), at, c_next, h_next


class Seq2Seq(chainer.Chain):

    def __init__(self, all_vocab_size, emotion_vocab_size, feature_num, hidden_num, batch_size, gpu_flg):
        """
        :param all_vocab_size: input vocab size
        :param emotion_vocab_size: input emotion vocab size
        :param feature_num: size of feature layer (embed layer)
        :param hidden_num: size of hidden layer
        :return:
        """
        global xp
        xp = cuda.cupy if gpu_flg >= 0 else np

        self.all_vocab_size = all_vocab_size
        self.hidden_num = hidden_num
        self.batch_size = batch_size
        self.c_batch = Variable(xp.zeros((batch_size, self.hidden_num), dtype=xp.float32))  # cell Variable
        self.h_batch = Variable(xp.zeros((batch_size, self.hidden_num), dtype=xp.float32))  # hidden Variable

        super(Seq2Seq, self).__init__(
            enc=Encoder(all_vocab_size, feature_num, hidden_num, batch_size),               # encoder
            dec=Decoder(all_vocab_size, emotion_vocab_size, feature_num, hidden_num)        # decoder
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

    def decode(self, input_id, teacher_id, word_th, train=True):
        """
        :param input_id: batch of word ID by output of decoder
        :param teacher_id : batch of correct ID
        :param word_th : batch of correct at label
        :param train: True or false
        :return: decoded embed vector
        """
        batch_word = chainer.Variable(xp.array(input_id, dtype=xp.int32))
        predict_mat, predict_at, self.c_batch, self.h_batch = self.dec(batch_word, self.c_batch, self.h_batch, train=train)
        if train:
            t = xp.array(teacher_id, dtype=xp.int32)
            t = chainer.Variable(t)

            predict_ids = xp.argmax(predict_mat.data, axis=1)
            correct_at = xp.zeros((1, predict_ids.shape[0]), dtype=xp.float32)
            for ind in range(predict_ids.shape[0]):
                # right answer
                if predict_ids[ind] < word_th and teacher_id[ind] < word_th:
                    correct_at[0, ind] = 1.0
                elif predict_ids[ind] > word_th and teacher_id[ind] > word_th:
                    correct_at[0, ind] = 1.0
                # wrong answer
                else:
                    correct_at[0, ind] = 0.0
            correct_at = chainer.Variable(correct_at.reshape(predict_ids.shape[0], 1))
            at_loss = -F.sum(F.log(predict_at) * correct_at)
            # if at_loss.data > 0:
            #     print(at_loss.data)
            return F.softmax_cross_entropy(predict_mat, t) + at_loss, predict_mat
        else:
            return predict_mat

    def initialize(self):
        self.c_batch = Variable(xp.zeros((self.batch_size, self.hidden_num), dtype=xp.float32))
        self.h_batch = Variable(xp.zeros((self.batch_size, self.hidden_num), dtype=xp.float32))

    def one_encode(self, src_text, train=False):
        """
        :param src_text: input text embed id ex.) [ 1, 0 ,14 ,5 ]
        :param train : True or False
        :return: context vector (hidden vector)
        """
        for word in src_text:
            word = chainer.Variable(xp.array([word], dtype=xp.int32))
            self.c_batch, self.h_batch = self.enc(word, self.c_batch, self.h_batch, train=train)

    def one_decode(self, input_id, teacher_id, correct_at, train=False):
        """
        :param input_id:
        :param teacher_id : embed id ( teacher's )
        :param correct_at:
        :param train: True or false
        :return: decoded embed vector
        """
        word = chainer.Variable(xp.array([input_id], dtype=xp.int32))
        predict_vec, predict_at, self.c_batch, self.h_batch = self.dec(word, self.c_batch, self.h_batch, train=train)
        if train:
            # TODO
            pass
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
            predict_vec = self.one_decode(input_id=word_id, teacher_id=None, correct_at=None, train=False)
            word = id2word[xp.argmax(predict_vec.data)]     # choose word_ID which has the highest probability
            word_id = word2id[word]
            if word == "<eos>":
                break
            sentence = sentence + word + " "
        return sentence

    def initial_state_function(self, X):
        self.initialize()
        self.one_encode(X, train=False)
        return self.c_batch, self.h_batch

    def generate_function(self, Y_tm1, state_tm1):
        state_t = []
        for index, (w_id, state) in enumerate(zip(Y_tm1, state_tm1)):
            self.c_batch = state[0]
            self.h_batch = state[1]
            predict_vec = self.one_decode(input_id=w_id, teacher_id=None, correct_at=None, train=False)
            state_t.append((self.c_batch, self.h_batch))
            if index == 0:
                p_t = predict_vec.data
            else:
                p_t = np.vstack((p_t, predict_vec.data))
        return state_t, p_t

    @staticmethod
    def beam_search(initial_state_function, generate_function, X, start_id, end_id,
                    beam_width=4, num_hypotheses=5, max_length=15):
        """
        Beam search for neural network sequence to sequence (encoder-decoder) models.

        :param initial_state_function: A function that takes X as input and returns state (2-dimensional numpy array with 1 row
                                       representing decoder recurrent layer state - currently supports only one recurrent layer).
        :param generate_function: A function that takes Y_tm1 (1-dimensional numpy array of token indices in decoder vocabulary
                                  generated at previous step) and state_tm1 (2-dimensional numpy array of previous step
                                  decoder recurrent layer states) as input.
                                  Returns state_t (2-dimensional numpy array of current step decoder recurrent
                                  layer states), p_t (2-dimensional numpy array of decoder softmax outputs).
        :param X: List of input token indices in encoder vocabulary.
        :param start_id: Index of <start sequence> token in decoder vocabulary.
        :param end_id: Index of <end sequence> token in decoder vocabulary.
        :param beam_width: Beam size. Default 4.
        :param num_hypotheses: Number of hypotheses to generate. Default 1.
        :param max_length: Length limit for generated sequence. Default 50.
        """

        # 入力データのタイプチェック
        if isinstance(X, list) or X.ndim == 1:
            X = np.array([X], dtype=np.int32).T
        assert X.ndim == 2 and X.shape[1] == 1, "X should be a column array with shape (input-sequence-length, 1)"

        # encode
        next_fringe = [Node(parent=None, state=initial_state_function(X), value=start_id, cost=0.0)]
        hypotheses = []

        for _ in range(max_length):

            # 予測候補単語リストの入替え(next_fringe => fringe にお引越し)
            fringe = []
            for n in next_fringe:
                if n.value == end_id:
                    hypotheses.append(n)
                else:
                    fringe.append(n)

            # 終了条件
            if not fringe:
                break

            # hypothesis同士の比較 (hypothesis数が上限値を超えている場合 => costが高いものから消去)
            if len(hypotheses) > num_hypotheses:
                sort_hypotheses = sorted([(hypothesis.to_cost_score(), hypothesis) for hypothesis in hypotheses],
                                         key=lambda x: x[0])[:num_hypotheses]
                hypotheses.clear()
                for tup_hypothesis in sort_hypotheses:
                    hypotheses.append(tup_hypothesis[1])

            Y_tm1 = [n.value for n in fringe]
            state_tm1 = [n.state for n in fringe]
            state_t, p_t = generate_function(Y_tm1, state_tm1)  # state_t: decの内部状態群, p_t: 各行にpredict_vec(単語次元)が入った行列
            Y_t = np.argsort(p_t, axis=1)[:, -beam_width:]      # Y_t: 大きい値上位beam幅件の配列番号リスト
            next_fringe = []
            for Y_t_n, p_t_n, state_t_n, n in zip(Y_t, p_t, state_t, fringe):
                Y_nll_t_n = -F.log_softmax(np.array([p_t_n])).data[:, Y_t_n][0, :]
                for y_t_n, y_nll_t_n in zip(Y_t_n, Y_nll_t_n):
                    n_new = Node(parent=n, state=state_t_n, value=y_t_n, cost=y_nll_t_n)
                    next_fringe.append(n_new)

            # 全コストを計算する場合
            next_fringe = sorted(next_fringe, key=lambda n: n.to_cost_score())[:beam_width]

        # 最終的なsorting
        hypotheses.sort(key=lambda n: n.to_cost_score())   # sentence全体のcostを計算したい場合
        return hypotheses[:num_hypotheses]


class Node(object):
    """
    ステップ毎の出力データの格納クラス
    """
    def __init__(self, parent, state, value, cost):
        super(Node, self).__init__()
        self.value = value
        self.parent = parent  # parent Node, None for root
        self.state = state  # recurrent layer hidden state
        self.cum_cost = parent.cum_cost + cost if parent else cost  # e.g. -log(p) of sequence up to current node (including)
        self.length = 1 if parent is None else parent.length + 1
        self._sequence = None

    def to_cost_score(self):
        cost = 0
        current_node = self
        while current_node:
            cost += current_node.cum_cost
            current_node = current_node.parent
        return cost

    def to_sequence(self):
        # Return sequence of nodes from root to current node.
        if not self._sequence:
            self._sequence = []
            current_node = self
            while current_node:
                self._sequence.insert(0, current_node)
                current_node = current_node.parent
        return self._sequence

    def to_sequence_of_values(self):
        return [s.value for s in self.to_sequence()]

    def to_sequence_of_extras(self):
        return [s.extras for s in self.to_sequence()]
