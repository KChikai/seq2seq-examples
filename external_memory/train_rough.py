# -*- coding:utf-8 -*-
"""
speaker model用プレトレイン
（大量のトレーニング文で学習を行う）
"""

import os
os.environ["CHAINER_TYPE_CHECK"] = "0"

import pickle
import argparse
import numpy as np
import chainer
from chainer import cuda, optimizers, serializers
from external_memory.tuning_util import JaConvCorpus, ConvCorpus
from external_memory.external_seq2seq import Seq2Seq
from setting_param import EPOCH, FEATURE_NUM, HIDDEN_NUM, LABEL_NUM, LABEL_EMBED, BATCH_NUM


# parse command line args
parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', default='./data/pair_corpus.txt', type=str, help='Data file directory')
parser.add_argument('--gpu', '-g', default='-1', type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=EPOCH, type=int, help='number of epochs to learn')
parser.add_argument('--feature_num', '-f', default=FEATURE_NUM, type=int, help='dimension of feature layer')
parser.add_argument('--hidden_num', '-hi', default=HIDDEN_NUM, type=int, help='dimension of hidden layer')
parser.add_argument('--label_num', '-ln', default=LABEL_NUM, type=int, help='dimension of label layer')
parser.add_argument('--label_embed', '-le', default=LABEL_EMBED, type=int, help='dimension of label embed layer')
parser.add_argument('--batchsize', '-b', default=BATCH_NUM, type=int, help='learning minibatch size')
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


def remove_extra_padding(batch_list):
    """
    remove extra padding
    """
    remove_row = []
    for i in range(len(batch_list))[::-1]:
        if sum(batch_list[i]) == -1 * len(batch_list[i]):
            remove_row.append(i)
        else:
            break
    return np.delete(batch_list, remove_row, axis=0)


def main():

    ###########################
    #### create dictionary ####
    ###########################

    if os.path.exists('./data/corpus/dictionary.dict'):
        if args.lang == 'ja':
            corpus = JaConvCorpus(file_path=None, batch_size=batchsize, size_filter=True)
        else:
            corpus = ConvCorpus(file_path=None, batch_size=batchsize, size_filter=True)
        corpus.load(load_dir='./data/corpus/')
    else:
        if args.lang == 'ja':
            corpus = JaConvCorpus(file_path=data_file, batch_size=batchsize, size_filter=True)
        else:
            corpus = ConvCorpus(file_path=data_file, batch_size=batchsize, size_filter=True)
        corpus.save(save_dir='./data/corpus/')
    print('Vocabulary Size (number of words) :', len(corpus.dic.token2id))

    print('Emotion size: ', len(corpus.emotion_set))
    ma = 0
    mi = 999999
    for word in corpus.emotion_set:
        wid = corpus.dic.token2id[word]
        if wid > ma:
            ma = wid
        if wid < mi:
            mi = wid
    # print(corpus.dic.token2id['<start>'], corpus.dic.token2id['<eos>'], corpus.dic.token2id['happy'], mi, ma)
    word_threshold = mi

    ######################
    #### create model ####
    ######################

    model = Seq2Seq(all_vocab_size=len(corpus.dic.token2id), emotion_vocab_size=len(corpus.emotion_set),
                    feature_num=feature_num, hidden_num=hidden_num, batch_size=batchsize, gpu_flg=args.gpu)
    if args.gpu >= 0:
        model.to_gpu()
    optimizer = optimizers.Adam(alpha=0.001)
    optimizer.setup(model)
    #optimizer.add_hook(chainer.optimizer.GradientClipping(5))
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    ##########################
    #### create ID corpus ####
    ##########################

    input_mat = []
    output_mat = []
    max_input_ren = max_output_ren = 0
    for input_text, output_text in zip(corpus.posts, corpus.cmnts):

        # convert to list
        input_text.reverse()                                        # 入力を反転させるかどうか
        # input_text.insert(0, corpus.dic.token2id["<eos>"])        # 入力の最初にeosを挿入
        output_text.append(corpus.dic.token2id["<eos>"])            # 出力の最後にeosを挿入

        # update max sentence length
        max_input_ren = max(max_input_ren, len(input_text))
        max_output_ren = max(max_output_ren, len(output_text))

        input_mat.append(input_text)
        output_mat.append(output_text)

    # padding (文末にパディングを挿入する)
    for li in input_mat:
        insert_num = max_input_ren - len(li)
        for _ in range(insert_num):
            li.append(corpus.dic.token2id['<pad>'])
    for li in output_mat:
        insert_num = max_output_ren - len(li)
        for _ in range(insert_num):
            li.append(corpus.dic.token2id['<pad>'])

    # create batch matrix
    input_mat = np.array(input_mat, dtype=np.int32).T
    output_mat = np.array(output_mat, dtype=np.int32).T

    # create correct_at matrix
    correct_at_mat = np.array(output_mat, dtype=np.float32)
    for r_index, row in enumerate(correct_at_mat):
        for c_index, w_id in enumerate(row):
            if w_id < word_threshold:
                correct_at_mat[r_index, c_index] = 0.0
            else:
                correct_at_mat[r_index, c_index] = 1.0

    with open('./data/corpus/input_mat.pkl', 'wb') as f:
        pickle.dump(input_mat, f)
    with open('./data/corpus/output_mat.pkl', 'wb') as f:
        pickle.dump(output_mat, f)

    # separate corpus into Train and Test (今回はテストしない)
    train_input_mat = input_mat
    train_output_mat = output_mat

    #############################
    #### train seq2seq model ####
    #############################

    accum_loss = 0
    train_loss_data = []
    for num, epoch in enumerate(range(n_epoch)):
        total_loss = 0
        batch_num = 0
        perm = np.random.permutation(len(corpus.posts))

        # for training
        for i in range(0, len(corpus.posts), batchsize):

            # select batch data
            input_batch = remove_extra_padding(train_input_mat[:, perm[i:i + batchsize]])
            output_batch = remove_extra_padding(train_output_mat[:, perm[i:i + batchsize]])
            correct_at_batch = correct_at_mat[:, perm[i:i + batchsize]]

            # Encode a sentence
            model.initialize()                     # initialize cell
            model.encode(input_batch, train=True)  # encode (output: hidden Variable)

            # Decode from encoded context
            end_batch = xp.array([corpus.dic.token2id["<start>"] for _ in range(batchsize)])
            first_words = output_batch[0]
            #correct_at = chainer.Variable(xp.array(correct_at_batch[0], dtype=xp.float32).reshape(batchsize, 1))
            # correct_at = chainer.Variable(xp.array([[0] for i in range(batchsize)], dtype=xp.float32))
            loss, predict_mat = model.decode(end_batch, first_words, word_threshold, train=True)
            next_ids = first_words
            accum_loss += loss
            for r_ind, w_ids in enumerate(output_batch[1:]):
                #correct_at = chainer.Variable(xp.array(correct_at_batch[r_ind+1], dtype=xp.float32).reshape(batchsize, 1))
                # correct_at = chainer.Variable(xp.array([[0] if i < word_threshold else [1] for i in w_ids], dtype=xp.float32))
                # for index, f in enumerate(correct_at.data):
                #     if f != 0:
                #         print(correct_at.data)
                #         print(corpus.dic[w_ids[index]])
                #         break
                loss, predict_mat = model.decode(next_ids, w_ids, word_threshold, train=True)
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

        # save model and optimizer
        if (epoch + 1) % 5 == 0:
            print('-----', epoch + 1, ' times -----')
            print('save the model and optimizer')
            serializers.save_hdf5('data/' + str(epoch) + '.model', model)
            serializers.save_hdf5('data/' + str(epoch) + '.state', optimizer)

    # save loss data
    with open('./data/loss_train_data.pkl', 'wb') as f:
        pickle.dump(train_loss_data, f)


if __name__ == "__main__":
    main()