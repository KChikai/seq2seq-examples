# -*- coding:utf-8 -*-

"""
低頻度語を分散表現でカバーするモデル．
wikipedia等の学習モデルを事前に用意しておく．
モデル形式はgensimでのロードを可能にする必要がある．
"""

import re
import pickle
import gensim
from gensim import corpora


# the path of the word2vec model
W2V_MODEL_PATH = './data/neo_model.vec'


class JaConvCorpus:
    """
    Dictionary Class for Japanese
    とりあえず，入力文の最後に感情値が付随しているケースを考える．
    """
    def __init__(self, file_path, batch_size=100, size_filter=False):
        self.rough_posts = []
        self.rough_cmnts = []
        self.fine_posts = []
        self.fine_cmnts = []
        self.dic = None

        if file_path is not None:
            self._construct_dict(file_path, batch_size, size_filter)

    def _construct_dict(self, file_path, batch_size, size_filter):
        # define sentence and corpus size
        max_length = 20
        batch_size = batch_size

        # preprocess (for rough data)
        rough_posts = []
        rough_cmnts = []
        pattern = '(.+?)(\t)(.+?)(\n|\r\n)'
        r = re.compile(pattern)
        for index, line in enumerate(open(file_path, 'r', encoding='utf-8')):
            m = r.search(line)
            if m is not None:
                post = [word for word in m.group(1).split(' ')]
                cmnt = [word for word in m.group(3).split(' ')]
                if size_filter:
                    if len(post) <= max_length and len(cmnt) <= max_length:
                        rough_posts.append(post)
                        rough_cmnts.append(cmnt)
                else:
                    rough_posts.append(post)
                    rough_cmnts.append(cmnt)

        # cut corpus for a batch size (for training by rough data)
        remove_num = len(rough_posts) - (int(len(rough_posts) / batch_size) * batch_size)
        del rough_posts[len(rough_posts) - remove_num:]
        del rough_cmnts[len(rough_cmnts) - remove_num:]
        print(len(rough_posts), 'of rough pairs has been collected!')

        # preprocess (for fine data)
        fine_posts = []
        fine_cmnts = []
        for index, line in enumerate(open('./data/fine_pair_corpus.txt', 'r', encoding='utf-8')):
            m = r.search(line)
            if m is not None:
                post = [word for word in m.group(1).split(' ')]
                cmnt = [word for word in m.group(3).split(' ')]
                if size_filter:
                    if len(post) <= max_length and len(cmnt) <= max_length:
                        fine_posts.append(post)
                        fine_cmnts.append(cmnt)
                else:
                    fine_posts.append(post)
                    fine_cmnts.append(cmnt)

        # cut corpus for a batch size (for training by fine data)
        remove_num = len(fine_posts) - (int(len(fine_posts) / batch_size) * batch_size)
        del fine_posts[len(fine_posts) - remove_num:]
        del fine_cmnts[len(fine_cmnts) - remove_num:]
        print(len(fine_posts), 'of fine pairs has been collected!')

        # construct a whole dictionary
        self.dic = corpora.Dictionary(rough_posts + rough_cmnts + fine_posts + fine_cmnts, prune_at=None)
        self.dic.filter_extremes(no_below=2, no_above=1.0, keep_n=40000)    # remove low frequency words

        # add symbols
        self.dic.token2id['<start>'] = len(self.dic.token2id)
        self.dic.token2id['<eos>'] = len(self.dic.token2id)
        self.dic.token2id['<unk>'] = len(self.dic.token2id)
        self.dic.token2id['<pad>'] = -1

        # make ID corpus
        sim_th = 50
        model = gensim.models.KeyedVectors.load_word2vec_format(W2V_MODEL_PATH, binary=False)
        self.rough_posts = self._token_to_id(token_data=rough_posts, model=model, sim_th=sim_th)
        self.rough_cmnts = self._token_to_id(token_data=rough_cmnts, model=model, sim_th=sim_th)
        self.fine_posts = self._token_to_id(token_data=fine_posts, model=model, sim_th=sim_th)
        self.fine_cmnts = self._token_to_id(token_data=fine_cmnts, model=model, sim_th=sim_th)

    def _token_to_id(self, token_data, model, sim_th):
        """
        単語列コーパスをid化する．
        その際にword2vecを考慮したid化を行う．
        :param token_data: 単語列コーパス
        :param model: 分散表現モデル
        :param sim_th: モデルから類似度上位何件見るかの閾値
        :return:
        """
        all_word_num = 0
        replace_num = 0
        unk_dic_num = 0
        unk_w2v_num = 0

        corpus_id = []
        for text in token_data:
            text_ids = []
            for word in text:
                all_word_num += 1
                # 入力単語のIDがある場合
                if self.dic.token2id.get(word) is not None:
                    text_ids.append(self.dic.token2id.get(word))
                # ない場合（低頻度語）
                else:
                    # word2vec内に対象単語が存在する場合
                    try:
                        sim_words = model.most_similar(positive=[word], topn=sim_th)
                        for index, candidate_tuple in enumerate(sim_words):
                            # 辞書内の既存の単語として置き換えが可能の場合
                            if self.dic.token2id.get(candidate_tuple[0]) is not None:
                                replace_num += 1
                                # print(word, '=>', candidate_tuple[0], 'に置き換え成功！ ')
                                text_ids.append(self.dic.token2id.get(candidate_tuple[0]))
                                break
                            # 辞書内の単語と合致しなかった場合
                            if index == sim_th - 1:
                                unk_dic_num += 1
                                # print(word, 'の置き換え失敗しました．．．')
                                text_ids.append(self.dic.token2id['<unk>'])
                    # word2vec内に対象単語が存在しない場合
                    except KeyError:
                        unk_w2v_num += 1
                        # print(word, 'の置き換え候補がありませんでした．．．')
                        text_ids.append(self.dic.token2id['<unk>'])
            corpus_id.append(text_ids)
        print('全語彙数　　：', len(self.dic.token2id))
        print('全単語出現数：', all_word_num)
        print('置換え成功数：', replace_num)
        print('unk数出現数：', unk_dic_num + unk_w2v_num,
              '(辞書内単語不一致：', unk_dic_num, ', word2vec単語不一致：', unk_w2v_num, ')')
        with open('./data/log.txt', 'a', encoding='utf-8') as f:
            f.write('全語彙数　　：' + str(len(self.dic.token2id)) + '\n')
            f.write('全単語出現数：' + str(all_word_num) + '\n')
            f.write('置換え成功数：' + str(replace_num) + '\n')
            f.write('unk数出現数：' + str(unk_dic_num + unk_w2v_num) +
                    '(辞書内単語不一致：' + str(unk_dic_num) + ', word2vec単語不一致：' + str(unk_w2v_num) + ')' + '\n\n')

        return corpus_id

    def save(self, save_dir):
        self.dic.save(save_dir + 'dictionary.dict')
        with open(save_dir + 'rough_posts.list', 'wb') as f:
            pickle.dump(self.rough_posts, f)
        with open(save_dir + 'rough_cmnts.list', 'wb') as f:
            pickle.dump(self.rough_cmnts, f)
        with open(save_dir + 'fine_posts.list', 'wb') as f:
            pickle.dump(self.fine_posts, f)
        with open(save_dir + 'fine_cmnts.list', 'wb') as f:
            pickle.dump(self.fine_cmnts, f)

    def load(self, load_dir):
        self.dic = corpora.Dictionary.load(load_dir + 'dictionary.dict')
        with open(load_dir + 'rough_posts.list', 'rb') as f:
            self.rough_posts = pickle.load(f)
        with open(load_dir + 'rough_cmnts.list', 'rb') as f:
            self.rough_cmnts = pickle.load(f)
        with open(load_dir + 'fine_posts.list', 'rb') as f:
            self.fine_posts = pickle.load(f)
        with open(load_dir + 'fine_cmnts.list', 'rb') as f:
            self.fine_cmnts = pickle.load(f)
        print(len(self.rough_posts), 'of rough pairs has been collected!')
        print(len(self.fine_posts), 'of fine pairs has been collected!')


# test code
if __name__ == '__main__':
    corpus = JaConvCorpus(file_path='./data/pair_corpus.txt', batch_size=100, size_filter=True)
    corpus.save(save_dir='./data/corpus/')
