import os
import json
import sys
import pickle
import math
import numpy as np
import scipy
import scipy.sparse as sp
import torch
import random
import copy
from tqdm import tqdm
# from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from graph.file_overnight import *

PAD = 0
BOS = 1
EOS = 2
UNK = 3
stopwords_set = set(stopwords.words('english'))
stopwords_set.add('URL')


class Vocab:
    def __init__(self, vocab_file, emb_file, emb_size, use_pre_emb=False, vocab_size=50000):
        self.emb_size = emb_size
        self._word2id = {'[PADDING]': 0, '[START]': 1, '[END]': 2, '[OOV]': 3}
        self._id2word = ['[PADDING]', '[START]', '[END]', '[OOV]']
        self._wordcount = {'[PADDING]': 1, '[START]': 1, '[END]': 1, '[OOV]': 1}
        # self.lemmatizer = WordNetLemmatizer()
        if use_pre_emb:
            emb = self.load_glove(emb_file, self.emb_size)
            self.load_vocab(vocab_file, vocab_size=vocab_size, emb_size=emb_size, word_emb=emb)
            self.emb = np.array(self.emb, dtype=np.float32)
        else:
            self.emb = None
            self.load_vocab(vocab_file, emb_size=emb_size, vocab_size=vocab_size)
        self.voc_size = len(self._word2id)
        self.UNK_token = 3
        self.PAD_token = 0

    @staticmethod
    def load_glove(fname, emb_size):
        emb = {}
        for line in open(fname):
            tem = line.strip().split(' ')
            word = tem[0]
            vec = np.array([float(num) for num in tem[1:]], dtype=np.float32)
            if len(vec) != emb_size:
                #print('emb size', len(vec), vec)
                continue
            emb[word] = vec
        return emb

    def load_vocab(self, vocab_file, emb_size, word_emb=None, vocab_size=None):
        if word_emb is not None:
            self.emb = [np.zeros(emb_size) for _ in range(4)]
        words = []
        for line in open(vocab_file):
            try:
                word, count = line.strip().split(' ')
                words.append(word)
            except ValueError:
                print(line)
            if vocab_size > 0 and len(words) >= vocab_size:
                break
        for word in words:
            '''
            word_lem = self.lemmatizer.lemmatize(word)
            if word_emb is not None and (word in word_emb or word_lem in word_emb):
                self._word2id[word] = len(self._word2id)
                self._id2word.append(word)
                if word in word_emb:
                    self.emb.append(word_emb[word])
                else:
                    self.emb.append(word_emb[word_lem])
            '''
            if word_emb is not None and word in word_emb:
                self._word2id[word] = len(self._word2id)
                self._id2word.append(word)
                self.emb.append(word_emb[word])
            elif word_emb is None:
                self._word2id[word] = len(self._word2id)
                self._id2word.append(word)

        assert len(self._word2id) == len(self._id2word)

    def word2id(self, word):
        if word in self._word2id:
            return self._word2id[word]
        '''
        word_lem = self.lemmatizer.lemmatize(word)
        if word_lem in self._word2id:
            return self._word2id[word_lem]
        '''
        return self._word2id['[OOV]']

    def sent2id(self, sent, add_start=False, add_end=False):
        result = [self.word2id(word) for word in sent]
        if add_start:
            result = [self._word2id['[START]']] + result

        if add_end:
            result = result + [self._word2id['[END]']]
        return result

    def id2word(self, word_id):
        return self._id2word[word_id]

    def id2sent(self, sent_id):
        result = []
        for id in sent_id:
            if id == self._word2id['[END]']:
                break
            elif id == self._word2id['[PADDING]']:
                continue
            result.append(self._id2word[id])
        return result

    def sent2emb(self, sent_id):
        emb = np.zeros(self.emb_size)
        for word_id in sent_id:
            emb += self.emb[word_id]
        return emb


class DataLoader:
    def __init__(self, config, vocab, debug=False):
        self.debug = debug
        self.vocab = vocab
        self.graph = self.load_graph(config)
        self.adjs, self.train, self.dev, self.test = self.load_data(config)

    def load_graph(self, config):
        # core30, large70 = read_company_list()
        companies, tpx, prices, tweets = read_data(config)
        graph = Graph(companies, tpx, config)
        return graph

    def load_data(self, config):
        # graph, spans = pickle.load(open(config.data, 'rb'))
        train_span, dev_span, test_span = self.graph.extract_data_span()
        print('loading adj')
        adjs = self.graph.load_adjacency()
        # adjs = [self.normalize(adjs[0]), self.normalize(adjs[1])]
        adjs = [self.normalize(adj) for adj in adjs]
        print('loading datasets')
        return adjs, self.read_dataset(train_span, config), self.read_dataset(dev_span, config), self.read_dataset(
            test_span, config)

    def normalize(self, adj):
        adj = sp.coo_matrix(adj, shape=(len(adj), len(adj)), dtype=np.float32)
        adj = normalize(adj)
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        return adj

    def read_dataset(self, spans, config):
        result = []
        movement_num = 0
        for i in tqdm(range(len(spans))):
            span = spans[i]
            if self.debug and i >= 10:
                break
            if config.hierarchical:
                text, word_mask, sent_mask = span.pad_text_hierarchical(self.vocab, max_sent=config.max_sentence,
                                                                        max_len=config.max_text_len)
                text = [torch.from_numpy(np.array(node_text, dtype=np.long)) for node_text in text]
                word_mask = [torch.from_numpy(np.array(node_word_mask, dtype=np.float32)) for node_word_mask in
                             word_mask]
                sent_mask = torch.from_numpy(np.array(sent_mask, dtype=np.float32))
            else:
                text, word_mask = span.pad_text(self.vocab, max_len=config.max_text_len)
                text = torch.from_numpy(np.array(text, dtype=np.long))
                word_mask = torch.from_numpy(np.array(word_mask, dtype=np.float32))
                sent_mask = None
            bert_vec = torch.from_numpy(np.array(span.bert_vec, dtype=np.float32))
            span_nodes = torch.from_numpy(np.array(span.span_nodes, dtype=np.long))
            features = torch.from_numpy(np.array(span.span_features, dtype=np.float32))
            last_movement = torch.from_numpy(np.array(span.span_movement, dtype=np.long))
            tpx_movement = torch.from_numpy(np.array(span.span_tpx_movement, dtype=np.long))
            movement_mask = torch.from_numpy(np.array(span.span_movement_mask, dtype=np.float32))
            news_mask = torch.from_numpy(np.array(span.span_news_mask, dtype=np.float32))
            result.append((span_nodes, bert_vec, text, word_mask, sent_mask, features, last_movement, tpx_movement, movement_mask, news_mask,
                           span.movement_num))
            movement_num += sum(span.span_movement_mask)
        print('valid movement number', movement_num)

        return result


def normalize(adj):
    """Row-normalize sparse matrix"""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


if __name__ == '__main__':
    pass
