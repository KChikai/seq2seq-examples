# -*- coding:utf-8 -*-

import os
os.environ["CHAINER_TYPE_CHECK"] = "0"

import gc
import argparse
import numpy as np
from tuning_util import JaConvCorpus


# parse command line args
parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', default='./data/rough_pair_corpus.txt', type=str, help='Data file directory')
parser.add_argument('--batchsize', '-b', default=100, type=int, help='learning minibatch size')
parser.add_argument('--lang', '-l', default='ja', type=str, help='the choice of a language (Japanese "ja" or English "en" )')
args = parser.parse_args()
data_file = args.data
batchsize = args.batchsize


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

    ##########################
    #### create ID corpus ####
    ##########################

    input_mat = []
    output_mat = []
    batch_num = text_num = max_input_ren = max_output_ren = 0

    if not os.path.exists('./data/corpus/input_mat0.npy'):
        for input_text, output_text in zip(corpus.rough_posts, corpus.rough_cmnts):

            # convert to list
            input_text.reverse()                               # encode words in a reverse order
            input_text.insert(0, corpus.dic.token2id["<eos>"])
            output_text.append(corpus.dic.token2id["<eos>"])

            # update max sentence length
            max_input_ren = max(max_input_ren, len(input_text))
            max_output_ren = max(max_output_ren, len(output_text))

            input_mat.append(input_text)
            output_mat.append(output_text)
            batch_num += 1

            if batch_num % 10000 == 0:
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

                # save matrix and free memory
                print('save data ... number', text_num)
                np.save('./data/corpus/input_mat' + str(text_num) + '.npy', input_mat)
                np.save('./data/corpus/output_mat' + str(text_num) + '.npy', output_mat)
                text_num += 1
                del input_mat
                del output_mat
                gc.collect()
                input_mat = []
                output_mat = []

    else:
        print('You already have matrix files! '
              'If you remake new corpus, you should remove old files in "data/corpus" directory and run this script.')


if __name__ == '__main__':
    main()

    # check size
    # import glob
    # for index, text_name in enumerate(glob.glob('data/corpus/input_mat*')):
    #     batch_input_mat = np.load(text_name)
    #     print(batch_input_mat.shape)