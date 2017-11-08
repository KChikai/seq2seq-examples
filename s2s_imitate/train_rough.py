# -*- coding:utf-8 -*-
"""
Sample script of Sequence to Sequence model for ChatBot.
This is a train script for seq2seq.py
You can also use Batch and GPU.
args: --gpu (flg of GPU, if you want to use GPU, please write "--gpu 1")

単語次元：1024，隠れ層：2048
単語語彙数：25000
目的関数：Adam, 関数の初期化をエポック毎に行う

"""

import os
os.environ["CHAINER_TYPE_CHECK"] = "0"

import glob
import pickle
import argparse
import numpy as np
import chainer
from chainer import cuda, optimizers, serializers
from tuning_util import JaConvCorpus
from seq2seq import Seq2Seq


# parse command line args
parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', default='./data/rough_pair_corpus.txt', type=str, help='Data file directory')
parser.add_argument('--gpu', '-g', default='-1', type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=200, type=int, help='number of epochs to learn')
parser.add_argument('--feature_num', '-f', default=1024, type=int, help='dimension of feature layer')
parser.add_argument('--hidden_num', '-hi', default=2048, type=int, help='dimension of hidden layer')
parser.add_argument('--batchsize', '-b', default=100, type=int, help='learning minibatch size')
parser.add_argument('--testsize', '-t', default=1000, type=int, help='number of text for testing a model')
parser.add_argument('--lang', '-l', default='ja', type=str, help='the choice of a language (Japanese "ja" or English "en" )')
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


def main():

    ###########################
    #### create dictionary ####
    ###########################

    if os.path.exists('./data/corpus/dictionary.dict'):
        corpus = JaConvCorpus(file_path=None, batch_size=batchsize, size_filter=True)
        corpus.load(load_dir='./data/corpus/')
    else:
        corpus = JaConvCorpus(file_path=data_file, batch_size=batchsize, size_filter=True)
        corpus.save(save_dir='./data/corpus/')
    print('Vocabulary Size (number of words) :', len(corpus.dic.token2id))

    ######################
    #### create model ####
    ######################

    model = Seq2Seq(len(corpus.dic.token2id), feature_num=feature_num,
                    hidden_num=hidden_num, batch_size=batchsize, gpu_flg=args.gpu)
    if args.gpu >= 0:
        model.to_gpu()

    ##########################
    #### create ID corpus ####
    ##########################

    test_input_mat = []
    test_output_mat = []
    train_input_mats = []
    train_output_mats = []

    if not os.path.exists('./data/corpus/input_mat0.npy'):
        print("You don't have any input matrix. You should run 'preprocess.py' before you run this script.")
        raise ValueError
    else:
        for index, text_name in enumerate(glob.glob('data/corpus/input_mat*')):
            batch_input_mat = np.load(text_name)
            if index == 0:
                # separate corpus into Train and Test
                perm = np.random.permutation(batch_input_mat.shape[1])
                test_input_mat = batch_input_mat[:, perm[0:0 + testsize]]
                train_input_mats.append(batch_input_mat[:, perm[testsize:]])
            else:
                train_input_mats.append(batch_input_mat)
        for index, text_name in enumerate(glob.glob('data/corpus/output_mat*')):
            batch_output_mat = np.load(text_name)
            if index == 0:
                # separate corpus into Train and Test
                test_output_mat = batch_output_mat[:, perm[0:0 + testsize]]
                train_output_mats.append(batch_output_mat[:, perm[testsize:]])
            else:
                train_output_mats.append(batch_output_mat)

    list_of_references = []
    for text_ndarray in test_output_mat.T:
        reference = text_ndarray.tolist()
        references = [[w_id for w_id in reference if w_id is not -1]]
        list_of_references.append(references)

    #############################
    #### train seq2seq model ####
    #############################

    matrix_row_size = train_input_mats[0].shape[1] - testsize
    accum_loss = 0
    train_loss_data = []

    for num, epoch in enumerate(range(n_epoch)):
        total_loss = test_loss = batch_num = 0

        # initialize optimizer
        optimizer = optimizers.Adam(alpha=0.001)
        optimizer.setup(model)
        # optimizer.add_hook(chainer.optimizer.GradientClipping(5))
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

        # for training by each corpus matrix
        for mat_index in range(len(train_input_mats)):
            perm = np.random.permutation(matrix_row_size)

            # by each batch size
            for i in range(0, matrix_row_size, batchsize):

                # select batch data
                input_batch = train_input_mats[mat_index][:, perm[i:i + batchsize]]
                output_batch = train_output_mats[mat_index][:, perm[i:i + batchsize]]

                # Encode a sentence
                model.initialize()                     # initialize cell
                model.encode(input_batch, train=True)  # encode (output: hidden Variable)

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
                model.cleargrads()
                accum_loss.backward()
                optimizer.update()
                total_loss += float(accum_loss.data)
                print('Epoch: ', num, 'Matrix_num: ', mat_index, 'Batch_num', batch_num,
                      'batch loss: {:.2f}'.format(float(accum_loss.data)))
                batch_num += 1
                accum_loss = 0

        # save model and optimizer
        if (epoch + 1) % 5 == 0:
            print('-----', epoch + 1, ' times -----')
            print('save the model and optimizer')
            serializers.save_hdf5('data/' + str(epoch) + '_rough.model', model)
            serializers.save_hdf5('data/' + str(epoch) + '_rough.state', optimizer)

        # display the on-going status
        print('Epoch: ', num,
              'Train loss: {:.2f}'.format(total_loss))
        train_loss_data.append(float(total_loss / batch_num))

    # save loss data
    with open('./data/rough_loss_train_data.pkl', 'wb') as f:
        pickle.dump(train_loss_data, f)


if __name__ == "__main__":
    main()