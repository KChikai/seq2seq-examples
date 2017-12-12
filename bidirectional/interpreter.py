# -*- coding:utf-8 -*-


"""
タグをencoderの最後に入力したモデル用インタプリタ
入力文の後に半角＋数字を入力することでラベルを挿入する
"""

import os
os.environ["CHAINER_TYPE_CHECK"] = "0"

import argparse
import unicodedata
import pickle
import numpy as np
import matplotlib.pyplot as plt
from nltk import word_tokenize
from chainer import serializers, cuda
from util import ConvCorpus, JaConvCorpus
from s2s_bidirectional.seq2seq import Seq2Seq


# path info
DATA_DIR = './data/corpus/'
MODEL_PATH = './data/19.model'
TRAIN_LOSS_PATH = './data/loss_train_data.pkl'
TEST_LOSS_PATH = './data/loss_test_data.pkl'
BLEU_SCORE_PATH = './data/bleu_score_data.pkl'
WER_SCORE_PATH = './data/wer_score_data.pkl'

# parse command line args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default='-1', type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--feature_num', '-f', default=256, type=int, help='dimension of feature layer')
parser.add_argument('--hidden_num', '-hi', default=512, type=int, help='dimension of hidden layer')
parser.add_argument('--bar', '-b', default='1', type=int, help='whether to show the graph of loss values or not')
parser.add_argument('--lang', '-l', default='en', type=str, help='the choice of a language (Japanese "ja" or English "en" )')
args = parser.parse_args()

# GPU settings
gpu_device = args.gpu
if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(gpu_device).use()


def parse_ja_text(text):
    """
    Function to parse Japanese text.
    :param text: string: sentence written by Japanese
    :return: list: parsed text
    """
    import MeCab
    mecab = MeCab.Tagger("mecabrc")
    mecab.parse('')

    # list up noun
    mecab_result = mecab.parseToNode(text)
    parse_list = []
    while mecab_result is not None:
        if mecab_result.surface != "":
            parse_list.append(unicodedata.normalize('NFKC', mecab_result.surface).lower())
        mecab_result = mecab_result.next

    return parse_list


def interpreter(data_path, model_path):
    """
    Run this function, if you want to talk to seq2seq model.
    if you type "exit", finish to talk.
    :param data_path: the path of corpus you made model learn
    :param model_path: the path of model you made learn
    :return:
    """
    # call dictionary class
    if args.lang == 'en':
        corpus = ConvCorpus(file_path=None)
        corpus.load(load_dir=data_path)
    elif args.lang == 'ja':
        corpus = JaConvCorpus(file_path=None)
        corpus.load(load_dir=data_path)
    else:
        print('You gave wrong argument to this system. Check out your argument about languages.')
        raise ValueError
    print('Vocabulary Size (number of words) :', len(corpus.dic.token2id))
    print('')

    # rebuild seq2seq model
    model = Seq2Seq(len(corpus.dic.token2id), feature_num=args.feature_num,
                    hidden_num=args.hidden_num, batch_size=1, gpu_flg=args.gpu)
    serializers.load_hdf5(model_path, model)

    # run conversation system
    print('The system is ready to run, please talk to me!')
    print('( If you want to end a talk, please type "exit". )')
    print('')
    while True:
        print('>> ', end='')
        sentence = input()
        if sentence == 'exit':
            print('See you again!')
            break

        input_vocab = [unicodedata.normalize('NFKC', word.lower()) for word in word_tokenize(sentence)]
        input_vocab_rev = input_vocab[::-1]

        # convert word into ID
        input_sentence = [corpus.dic.token2id[word] for word in input_vocab if not corpus.dic.token2id.get(word) is None]
        input_sentence_rev = [corpus.dic.token2id[word] for word in input_vocab_rev if not corpus.dic.token2id.get(word) is None]

        model.initialize()          # initialize cell
        sentence = model.generate(input_sentence, input_sentence_rev, sentence_limit=len(input_sentence) + 30,
                                  word2id=corpus.dic.token2id, id2word=corpus.dic)
        print("-> ", sentence)
        print('')


def test_run(data_path, model_path, n_show=50):
    """
    Test function.
    Input is training data.
    Output have to be the sentence which is correct data in training phase.
    :return:
    """

    corpus = ConvCorpus(file_path=None)
    corpus.load(load_dir=data_path)

    print('Vocabulary Size (number of words) :', len(corpus.dic.token2id))
    print('')

    # rebuild seq2seq model
    model = Seq2Seq(len(corpus.dic.token2id), feature_num=args.feature_num,
                    hidden_num=args.hidden_num, batch_size=1, gpu_flg=args.gpu)
    serializers.load_hdf5(model_path, model)

    # run an interpreter
    for num, input_sentence in enumerate(corpus.posts):
        id_sequence = input_sentence.copy()
        input_sentence_rev = input_sentence[::-1]

        model.initialize()  # initialize cell
        sentence = model.generate(input_sentence, input_sentence_rev, sentence_limit=len(input_sentence) + 30,
                                  word2id=corpus.dic.token2id, id2word=corpus.dic)
        print("teacher : ", " ".join([corpus.dic[w_id] for w_id in id_sequence]))
        print("correct :", " ".join([corpus.dic[w_id] for w_id in corpus.cmnts[num]]))
        print("-> ", sentence)
        print('')

        if num == n_show:
            break


def show_chart(train_loss_path, test_loss_path):
    """
    Show the graph of Losses for each epochs
    """
    with open(train_loss_path, mode='rb') as f:
        train_loss_data = np.array(pickle.load(f))
    #with open(test_loss_path, mode='rb') as f:
    #    test_loss_data = np.array(pickle.load(f))
    row = len(train_loss_data)
    loop_num = np.array([i + 1 for i in range(row)])
    plt.plot(loop_num, train_loss_data, label="Train Loss Value", color="gray")
    #plt.plot(loop_num, test_loss_data, label="Test Loss Value", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc=2)
    plt.title("Learning Rate of Seq2Seq Model")
    plt.show()


def show_bleu_chart(bleu_score_path):
    """
    Show the graph of BLEU for each epochs
    """
    with open(bleu_score_path, mode='rb') as f:
        bleu_score_data = np.array(pickle.load(f))
    row = len(bleu_score_data)
    loop_num = np.array([i + 1 for i in range(row)])
    plt.plot(loop_num, bleu_score_data, label="BLUE score", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("BLEU")
    plt.legend(loc=2)
    plt.title("BLEU score of Seq2Seq Model")
    plt.show()


def show_wer_chart(wer_score_path):
    """
    Show the graph of WER for each epochs
    """
    with open(wer_score_path, mode='rb') as f:
        wer_score_data = np.array(pickle.load(f))
    row = len(wer_score_data)
    loop_num = np.array([i + 1 for i in range(row)])
    plt.plot(loop_num, wer_score_data, label="WER score", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("WER")
    plt.legend(loc=2)
    plt.title("WER score of Seq2Seq Model")
    plt.show()


if __name__ == '__main__':
    interpreter(DATA_DIR, MODEL_PATH)
    test_run(DATA_DIR, MODEL_PATH)
    if args.bar:
        show_chart(TRAIN_LOSS_PATH, TEST_LOSS_PATH)
        #show_bleu_chart(BLEU_SCORE_PATH)
        #show_wer_chart(WER_SCORE_PATH)
