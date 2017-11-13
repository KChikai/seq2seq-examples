# -*- coding:utf-8 -*-

import re
import pickle
import unicodedata
from gensim import corpora
from nltk import word_tokenize


def to_words(sentence):
    sentence_list = [re.sub(r"(\w+)(!+|\?+|…+|\.+|,+|~+)", r"\1", word) for word in sentence.split(' ')]
    return sentence_list


def is_english(string):
    for ch in string:
        try:
            name = unicodedata.name(ch)
        except ValueError:
            return False
        if "CJK UNIFIED" in name or "HIRAGANA" in name or "KATAKANA" in name:
            return False
    return True


class ConvCorpus:
    """
    Dictionary class for English
    """
    def __init__(self, file_path, batch_size=100, size_filter=False):
        self.posts = []
        self.cmnts = []
        self.dic = None

        if file_path is not None:
            self._construct_dict(file_path, batch_size, size_filter)

    def _construct_dict(self, file_path, batch_size, size_filter):
        # define sentence and corpus size
        max_length = 20
        batch_size = batch_size

        # preprocess
        posts = []
        cmnts = []
        pattern = '(.+?)(\t)(.+?)(\n|\r\n)'
        r = re.compile(pattern)
        for index, line in enumerate(open(file_path, 'r', encoding='utf-8')):
            m = r.search(line)
            if m is not None:
                if is_english(m.group(1) + m.group(3)):
                    post = [unicodedata.normalize('NFKC', word.lower()) for word in word_tokenize(m.group(1))]
                    cmnt = [unicodedata.normalize('NFKC', word.lower()) for word in word_tokenize(m.group(3))]
                    if size_filter:
                        if len(post) <= max_length and len(cmnt) <= max_length:
                            posts.append(post)
                            cmnts.append(cmnt)
                    else:
                        posts.append(post)
                        cmnts.append(cmnt)

        # cut corpus for a batch size
        remove_num = len(posts) - (int(len(posts) / batch_size) * batch_size)
        del posts[len(posts)-remove_num:]
        del cmnts[len(cmnts)-remove_num:]
        # print(len(posts), 'of pairs has been collected!')

        # construct dictionary
        self.dic = corpora.Dictionary(posts + cmnts, prune_at=None)
        self.dic.filter_extremes(no_below=1, no_above=1.0, keep_n=10000)
        print(len(self.dic))
        # add symbols
        self.dic.token2id['<start>'] = len(self.dic.token2id)
        self.dic.token2id['<eos>'] = len(self.dic.token2id)
        self.dic.token2id['<unk>'] = len(self.dic.token2id)
        self.dic.token2id['<pad>'] = -1

        # make ID corpus
        self.posts = [[self.dic.token2id.get(word, self.dic.token2id['<unk>']) for word in post] for post in posts]
        self.cmnts = [[self.dic.token2id.get(word, self.dic.token2id['<unk>']) for word in cmnt] for cmnt in cmnts]

    def save(self, save_dir):
        self.dic.save(save_dir + 'dictionary.dict')
        with open(save_dir + 'posts.list', 'wb') as f:
            pickle.dump(self.posts, f)
        with open(save_dir + 'cmnts.list', 'wb') as f:
            pickle.dump(self.cmnts, f)

    def load(self, load_dir):
        self.dic = corpora.Dictionary.load(load_dir + 'dictionary.dict')
        with open(load_dir + 'posts.list', 'rb') as f:
            self.posts = pickle.load(f)
        with open(load_dir + 'cmnts.list', 'rb') as f:
            self.cmnts = pickle.load(f)
        # print(len(self.posts), 'of pairs has been collected!')


class JaConvCorpus:
    """
    Dictionary Class for Japanese
    """
    def __init__(self, file_path, batch_size=100, size_filter=False):
        self.posts = []
        self.cmnts = []
        self.dic = None

        if file_path is not None:
            self._construct_dict(file_path, batch_size, size_filter)

    def _construct_dict(self, file_path, batch_size, size_filter):
        # define sentence and corpus size
        max_length = 100
        batch_size = batch_size

        # preprocess
        posts = []
        cmnts = []
        pattern = '(.+?)(\t)(.+?)(\n|\r\n)'
        r = re.compile(pattern)
        for index, line in enumerate(open(file_path, 'r', encoding='utf-8')):
            m = r.search(line)
            if m is not None:
                post = [word for word in m.group(1).split('　')]
                cmnt = [word for word in m.group(3).split('　')]
                if size_filter:
                    if len(post) <= max_length and len(cmnt) <= max_length:
                        posts.append(post)
                        cmnts.append(cmnt)
                else:
                    posts.append(post)
                    cmnts.append(cmnt)

        # cut corpus for a batch size
        remove_num = len(posts) - (int(len(posts) / batch_size) * batch_size)
        del posts[len(posts) - remove_num:]
        del cmnts[len(cmnts) - remove_num:]
        # print(len(posts), 'of pairs has been collected!')

        # construct dictionary
        self.dic = corpora.Dictionary(posts, prune_at=None)
        self.dic.filter_extremes(no_below=2, no_above=1.0, keep_n=10000)    # remove low frequency words

        # add symbols
        self.dic.token2id['<start>'] = len(self.dic.token2id)
        self.dic.token2id['<eos>'] = len(self.dic.token2id)
        self.dic.token2id['<unk>'] = len(self.dic.token2id)
        self.dic.token2id['<pad>'] = -1

        # make ID corpus
        self.posts = [[self.dic.token2id.get(word, self.dic.token2id['<unk>']) for word in post] for post in posts]
        self.cmnts = [[self.dic.token2id.get(word, self.dic.token2id['<unk>']) for word in cmnt] for cmnt in cmnts]

    def save(self, save_dir):
        self.dic.save(save_dir + 'dictionary.dict')
        with open(save_dir + 'posts.list', 'wb') as f:
            pickle.dump(self.posts, f)
        with open(save_dir + 'cmnts.list', 'wb') as f:
            pickle.dump(self.cmnts, f)

    def load(self, load_dir):
        self.dic = corpora.Dictionary.load(load_dir + 'dictionary.dict')
        with open(load_dir + 'posts.list', 'rb') as f:
            self.posts = pickle.load(f)
        with open(load_dir + 'cmnts.list', 'rb') as f:
            self.cmnts = pickle.load(f)
        # print(len(self.posts), 'of pairs has been collected!')