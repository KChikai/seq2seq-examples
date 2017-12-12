# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
Extract conversation text from Cornell Movie Dialogs Corpus.
Output: pair_corpus.txt (like the following format)

<post> <TAB> <its reply>
<post> <TAB> <its reply>
<post> <TAB> <its reply> ...

"""
import ssl
import re
import os
import unicodedata
import zipfile
from urllib import request
from nltk import word_tokenize
from gensim import corpora
from util import is_english
ssl._create_default_https_context = ssl._create_unverified_context


def unzip():
    request.urlretrieve(
        'http://www.mpi-sws.org/~cristian/data/cornell_movie_dialogs_corpus.zip',
        './data/cornell_movie_dialogs_corpus.zip')
    with zipfile.ZipFile('./data/cornell_movie_dialogs_corpus.zip') as zf:
        for name in zf.namelist():
            (dirname, filename) = os.path.split(name)
            if dirname == 'cornell movie-dialogs corpus':
                zf.extract(name, './data/')
    os.rename('./data/cornell movie-dialogs corpus', './data/cornell_movie-dialogs_corpus')
    os.system("nkf -Ew  ./data/cornell_movie-dialogs_corpus/movie_lines.txt > "
              "./data/cornell_movie-dialogs_corpus/movie_lines_utf-8.txt")


def main(threshold=50):

    # collect the conversation data from Cornell Movie Corpus
    text_dic = {}
    for text in open('./data/cornell_movie-dialogs_corpus/movie_lines_utf-8.txt', 'r', encoding='utf-8'):
        seq_list = text.split(" +++$+++ ")
        if len(seq_list) == 5:
            text_dic[seq_list[0]] = seq_list[4].split('\n')[0]

    conv_id_list = []
    for text in open('./data/cornell_movie-dialogs_corpus/movie_conversations.txt', 'r', encoding='utf-8'):
        seq_list = text.split(" +++$+++ ")
        conv_list = seq_list[3].replace("'", '').replace("[", '').replace("]\n", '').split(", ")
        conv_id_list.append((conv_list[0], conv_list[1]))

    # remove html tags and the sentences containing '--' or '...'
    tag_pattern = r"<[^>]*?>"
    tag = re.compile(tag_pattern)
    tag2_pattern = r"\.\.\.|--"
    tag2 = re.compile(tag2_pattern)
    posts = []
    cmnts = []
    for tup in conv_id_list[:]:
        if text_dic.get(tup[0], False) and text_dic.get(tup[1], False):
            # remove html tags
            text_dic[tup[0]] = tag.sub("", text_dic[tup[0]])
            text_dic[tup[1]] = tag.sub("", text_dic[tup[1]])

            # remove the sentences containing '--' or '...'
            m1 = tag2.search(text_dic[tup[0]])
            m2 = tag2.search(text_dic[tup[1]])
            if m1 is not None or m2 is not None:
                del text_dic[tup[0]]
                del text_dic[tup[1]]
                conv_id_list.remove(tup)
            else:
                # remove the sentences containing multi byte words
                if is_english(text_dic[tup[0]] + text_dic[tup[1]]):
                    posts.append([unicodedata.normalize('NFKC', word.lower())
                                  for word in word_tokenize(text_dic[tup[0]])])
                    cmnts.append([unicodedata.normalize('NFKC', word.lower())
                                  for word in word_tokenize(text_dic[tup[1]])])
                else:
                    del text_dic[tup[0]]
                    del text_dic[tup[1]]
                    conv_id_list.remove(tup)
        else:
            # remove a pair of IDs which is not exist in conversation corpus
            conv_id_list.remove(tup)

    # remove the sentences having low frequency words
    corpus = corpora.Dictionary(posts + cmnts, prune_at=None)
    cut_freq = {freq + 1 for freq in range(threshold)}
    with open('data/pair_corpus.txt', 'w', encoding='utf-8') as f:
        for i in range(len(posts)):
            post = set([corpus.dfs[corpus.token2id[word]] for word in posts[i]])
            cmnt = set([corpus.dfs[corpus.token2id[word]] for word in cmnts[i]])
            if len(post & cut_freq) == 0 and len(cmnt & cut_freq) == 0:
                f.write(text_dic[conv_id_list[i][0]] + '\t' + text_dic[conv_id_list[i][1]] + '\n')


if __name__ == '__main__':
    if not os.path.exists('./data/cornell_movie-dialogs_corpus/'):
        unzip()
    main()