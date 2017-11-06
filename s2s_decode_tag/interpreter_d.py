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
import collections
import numpy as np
import matplotlib.pyplot as plt
from nltk import word_tokenize
from chainer import serializers, cuda
from sklearn.metrics.pairwise import cosine_similarity
from util import ConvCorpus, JaConvCorpus
from seq2seq_d import Seq2Seq


# path info
DATA_DIR = './data/corpus/'
MODEL_PATH = './data/149.model'
TRAIN_LOSS_PATH = './data/loss_train_data.pkl'
TEST_LOSS_PATH = './data/loss_test_data.pkl'
BLEU_SCORE_PATH = './data/bleu_score_data.pkl'
WER_SCORE_PATH = './data/wer_score_data.pkl'

# parse command line args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default='-1', type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--feature_num', '-f', default=128, type=int, help='dimension of feature layer')
parser.add_argument('--hidden_num', '-hi', default=216, type=int, help='dimension of hidden layer')
parser.add_argument('--label_num', '-ln', default=2, type=int, help='dimension of label layer')
parser.add_argument('--label_embed', '-le', default=128, type=int, help='dimension of label embed layer')
parser.add_argument('--bar', '-b', default='0', type=int, help='whether to show the graph of loss values or not')
parser.add_argument('--lang', '-l', default='ja', type=str, help='the choice of a language (Japanese "ja" or English "en" )')
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
                    hidden_num=args.hidden_num, label_num=args.label_num,
                    label_embed_num=args.label_embed, batch_size=1, gpu_flg=args.gpu)
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

        # convert to a list
        if args.lang == 'en':
            input_vocab = [unicodedata.normalize('NFKC', word.lower()) for word in word_tokenize(sentence)]
        elif args.lang == 'ja':
            input_vocab = parse_ja_text(sentence)
        else:
            print("Sorry, but your language is not supported...")
            raise ValueError

        # check a sentiment tag
        label_id = -1
        if len(input_vocab) == 0:
            print('caution: you donot set any words!)')
            pass
        elif input_vocab[-1] == '2':
            del input_vocab[-1]
            label_id = 1
        elif input_vocab[-1] == '1':
            del input_vocab[-1]
            label_id = 0
        else:
            print('caution: you donot set any sentiment tags!')
            break

        # input_vocab.reverse()
        # input_vocab.insert(0, "<eos>")

        # convert word into ID
        input_sentence = [corpus.dic.token2id[word] for word in input_vocab if not corpus.dic.token2id.get(word) is None]

        model.initialize()          # initialize cell
        sentence = model.generate(input_sentence, sentence_limit=len(input_sentence) + 30,
                                  word2id=corpus.dic.token2id, id2word=corpus.dic, label_id=label_id)
        print("-> ", sentence)
        print('')


def show_chart(train_loss_path, test_loss_path):
    """
    Show the graph of Losses for each epochs
    """
    with open(train_loss_path, mode='rb') as f:
        train_loss_data = np.array(pickle.load(f))
    with open(test_loss_path, mode='rb') as f:
        test_loss_data = np.array(pickle.load(f))
    row = len(train_loss_data)
    loop_num = np.array([i + 1 for i in range(row)])
    plt.plot(loop_num, train_loss_data, label="Train Loss Value", color="gray")
    plt.plot(loop_num, test_loss_data, label="Test Loss Value", color="green")
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


def show_heatmap(data_path, model_path):
    import seaborn
    import matplotlib
    seaborn.set()
    matplotlib.rc('font', family='sans-serif')

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
                    hidden_num=args.hidden_num, label_num=args.label_num,
                    label_embed_num=args.label_embed, batch_size=1, gpu_flg=args.gpu)
    serializers.load_hdf5(model_path, model)

    # sentiment matrix in the decoder
    sentiment_mat = model.dec.le.W.data
    cmap = seaborn.diverging_palette(220, 10, as_cmap=True)     # Generate a custom diverging colormap
    seaborn.heatmap(sentiment_mat, cmap=cmap, center=0,
                    linewidths=.5, xticklabels=False)   # square=True, cbar_kws={"orientation": "horizontal"})
    plt.xlabel("Dimension (=" + str(sentiment_mat.shape[1]) + ")")
    plt.ylabel("Sentiment")
    plt.savefig('./data/sentiment_matrix.png')

    # encoder embedding matrix
    encode_mat = model.enc.xe.W
    seaborn.heatmap(encode_mat, cmap=cmap, center=0,
                    linewidths=.5, xticklabels=False)   # square=True, cbar_kws={"orientation": "horizontal"})
    plt.xlabel("Dimension (=" + str(sentiment_mat.shape[1]) + ")")
    plt.ylabel("Sentiment")
    plt.savefig('./data/sentiment_matrix.png')


def calculate_embedding_vectors(data_path, model_path):

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
                    hidden_num=args.hidden_num, label_num=args.label_num,
                    label_embed_num=args.label_embed, batch_size=1, gpu_flg=args.gpu)
    serializers.load_hdf5(model_path, model)

    # get embedding vectors
    embed_mat = model.dec.ye.W.data
    sentiment_mat = model.dec.le.W.data
    neg_vec = np.array([sentiment_mat[0, :]])
    pos_vec = np.array([sentiment_mat[1, :]])

    # calculate cos similarity
    neg_sim_dic = {}
    pos_sim_dic = {}
    for i in range(embed_mat.shape[0]):
        word_vec = np.array([embed_mat[i, :]])
        neg_sim_dic[i] = cosine_similarity(word_vec, neg_vec)
        pos_sim_dic[i] = cosine_similarity(word_vec, pos_vec)

        # if cosine_similarity(word_vec, pos_vec) > cosine_similarity(word_vec, neg_vec):
        #     print('pos: ', corpus.dic[i])
        # elif cosine_similarity(word_vec, pos_vec) < cosine_similarity(word_vec, neg_vec):
        #     print('neg: ', corpus.dic[i])
        # else:
        #     print('???: ', corpus.dic[i])
        #     raise ValueError

    # sort in descending order
    neg_ordered = collections.OrderedDict(sorted(neg_sim_dic.items(), key=lambda x: x[1], reverse=True))
    pos_ordered = collections.OrderedDict(sorted(pos_sim_dic.items(), key=lambda x: x[1], reverse=True))

    # show TOP50 words
    print('------- The words which is similar to a NEGATIVE tag --------')
    for index, w_index in enumerate(neg_ordered):
        print(corpus.dic[w_index], ': ', neg_ordered[w_index][0, 0])
        if index == 49:
            break
    print('------- The words which is similar to a POSITIVE tag --------')
    for index, w_index in enumerate(pos_ordered):
        print(corpus.dic[w_index], ': ', pos_ordered[w_index][0, 0])
        if index == 49:
            break



if __name__ == '__main__':

    # interpreter(DATA_DIR, MODEL_PATH)
    # show_heatmap(DATA_DIR, MODEL_PATH)
    calculate_embedding_vectors(DATA_DIR, MODEL_PATH)

    # if args.bar:
    #     show_chart(TRAIN_LOSS_PATH, TEST_LOSS_PATH)
    #     show_bleu_chart(BLEU_SCORE_PATH)
    #     show_wer_chart(WER_SCORE_PATH)
