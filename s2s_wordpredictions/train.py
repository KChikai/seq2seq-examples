# -*- coding:utf-8 -*-
"""
Sample script of Sequence to Sequence model for ChatBot.
This is a train script for seq2seq.py
You can also use Batch and GPU.
args: --gpu (flg of GPU, if you want to use GPU, please write "--gpu 1")
"""

import os
os.environ["CHAINER_TYPE_CHECK"] = "0"

import pickle
import argparse
import nltk
import numpy as np
import chainer
from chainer import cuda, optimizers, serializers, Variable
from s2s_wordpredictions.util import ConvCorpus, JaConvCorpus
from s2s_wordpredictions.seq2seq import Seq2Seq
from wer import wer


# parse command line args
parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', default='./data/pair_corpus.txt', type=str, help='Data file directory')
parser.add_argument('--gpu', '-g', default='-1', type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=100, type=int, help='number of epochs to learn')
parser.add_argument('--feature_num', '-f', default=256, type=int, help='dimension of feature layer')
parser.add_argument('--hidden_num', '-hi', default=512, type=int, help='dimension of hidden layer')
parser.add_argument('--batchsize', '-b', default=10, type=int, help='learning minibatch size')
parser.add_argument('--testsize', '-t', default=1000, type=int, help='number of text for testing a model')
parser.add_argument('--lang', '-l', default='en', type=str, help='the choice of a language (Japanese "ja" or English "en" )')
args = parser.parse_args()

# GPU settings
gpu_device = args.gpu
if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(gpu_device).use()
xp = cuda.cupy if args.gpu >= 0 else np

data_file = args.data
n_epoch = args.epoch
feature_num = args.feature_num
hidden_num = args.hidden_num
batchsize = args.batchsize
testsize = args.testsize


def create_wp_batch(vocab_size, wp_lists):
    array = np.zeros((args.batchsize, vocab_size), dtype=xp.float32)
    for row, li in enumerate(wp_lists):
        for wid in li:
            array[row, wid] = 1.0
    return Variable(array)


def main():

    ###########################
    #### create dictionary ####
    ###########################

    if os.path.exists('./data/corpus/dictionary.dict'):
        if args.lang == 'ja':
            corpus = JaConvCorpus(file_path=None, batch_size=batchsize, size_filter=True)
        else:
            corpus = ConvCorpus(file_path=None, batch_size=batchsize)
        corpus.load(load_dir='./data/corpus/')
    else:
        if args.lang == 'ja':
            corpus = JaConvCorpus(file_path=data_file, batch_size=batchsize, size_filter=True)
        else:
            corpus = ConvCorpus(file_path=data_file, batch_size=batchsize)
        corpus.save(save_dir='./data/corpus/')
    print('Vocabulary Size (number of words) :', len(corpus.dic.token2id))

    ######################
    #### create model ####
    ######################

    model = Seq2Seq(vocab_size=len(corpus.dic.token2id), feature_num=feature_num,
                    hidden_num=hidden_num, batch_size=batchsize, gpu_flg=args.gpu)
    if args.gpu >= 0:
        model.to_gpu()
    optimizer = optimizers.Adam(alpha=0.001)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5))

    ##########################
    #### create ID corpus ####
    ##########################

    input_mat = []
    output_mat = []
    output_wp_mat = []
    max_input_ren = max_output_ren = 0
    for input_text, output_text in zip(corpus.posts, corpus.cmnts):

        # convert to list
        # input_text.reverse()
        # input_text.insert(0, corpus.dic.token2id["<eos>"])
        output_text.append(corpus.dic.token2id["<eos>"])

        # update max sentence length
        max_input_ren = max(max_input_ren, len(input_text))
        max_output_ren = max(max_output_ren, len(output_text))

        input_mat.append(input_text)
        output_mat.append(output_text)

        # create word prediction matrix
        wp = []
        for wid in output_text:
            if wid in wp:
                wp.append(wid)
        output_wp_mat.append(wp)

    # padding
    for li in input_mat:
        insert_num = max_input_ren - len(li)
        for _ in range(insert_num):
            li.insert(0, corpus.dic.token2id['<pad>'])
    for li in output_mat:
        insert_num = max_output_ren - len(li)
        for _ in range(insert_num):
            li.append(corpus.dic.token2id['<pad>'])

    # create batch matrix
    input_mat = np.array(input_mat, dtype=np.int32).T
    output_mat = np.array(output_mat, dtype=np.int32).T

    # separate corpus into Train and Test
    perm = np.random.permutation(len(corpus.posts))
    test_input_mat = input_mat[:, perm[0:0 + testsize]]
    test_output_mat = output_mat[:, perm[0:0 + testsize]]
    train_input_mat = input_mat[:, perm[testsize:]]
    train_output_mat = output_mat[:, perm[testsize:]]
    train_output_wp_mat = []
    for index in perm[testsize:]:
        train_output_wp_mat.append(output_wp_mat[index])

    #############################
    #### train seq2seq model ####
    #############################

    accum_loss = 0
    train_loss_data = []
    for num, epoch in enumerate(range(n_epoch)):
        total_loss = 0
        batch_num = 0
        perm = np.random.permutation(len(corpus.posts) - testsize)

        # for training
        for i in range(0, len(corpus.posts) - testsize, batchsize):

            # select batch data
            input_batch = train_input_mat[:, perm[i:i + batchsize]]
            output_batch = train_output_mat[:, perm[i:i + batchsize]]
            output_wp_batch = []
            for index in perm[i:i + batchsize]:
                output_wp_batch.append(train_output_wp_mat[index])
            output_wp_batch = create_wp_batch(vocab_size=len(corpus.dic.token2id),
                                              wp_lists=output_wp_batch)

            # Encode a sentence
            model.initialize()                     # initialize cell
            loss = model.encode(input_batch, output_wp_batch, train=True)
            accum_loss += loss

            # Decode from encoded context
            end_batch = xp.array([corpus.dic.token2id["<start>"] for _ in range(batchsize)])
            first_words = output_batch[0]
            loss, predict_mat = model.decode(end_batch, first_words, train=True)
            next_ids = first_words
            accum_loss += loss
            for w_ids in output_batch[1:]:
                loss, predict_mat = model.decode(next_ids, w_ids, train=True)
                next_ids = w_ids
                accum_loss += loss

            # learn model
            model.cleargrads()     # initialize all grad to zero
            accum_loss.backward()  # back propagation
            optimizer.update()
            total_loss += float(accum_loss.data)
            batch_num += 1
            print('Epoch: ', num, 'Batch_num', batch_num, 'batch loss: {:.2f}'.format(float(accum_loss.data)))
            accum_loss = 0

    # save loss data
    with open('./data/loss_train_data.pkl', 'wb') as f:
        pickle.dump(train_loss_data, f)


if __name__ == "__main__":
    main()