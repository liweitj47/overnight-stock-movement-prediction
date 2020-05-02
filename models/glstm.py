import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import models


class Attentive_Pooling(nn.Module):
    def __init__(self, hidden_size):
        super(Attentive_Pooling, self).__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, hidden_size)
        self.u = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, memory, query=None, mask=None):
        '''

        :param query:   (node, hidden)
        :param memory: (node, hidden)
        :param mask:
        :return:
        '''
        if query is None:
            h = torch.tanh(self.w_1(memory))  # node, hidden
        else:
            h = torch.tanh(self.w_1(memory) + self.w_2(query))
        score = torch.squeeze(self.u(h), -1)  # node,
        if mask is not None:
            score = score.masked_fill(mask.eq(0), -1e9)
        alpha = F.softmax(score, -1)  # node,
        s = torch.sum(torch.unsqueeze(alpha, -1) * memory, -2)
        return s


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.Tanh()

    def forward(self, input, adj):
        support = self.linear(input)
        output = self.activation(torch.sparse.mm(adj, support))
        return output


class SLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, relation_num):
        super(SLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.Wh = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.Wn = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.U = nn.Linear(input_size, hidden_size * 4, bias=False)
        self.V = nn.Linear(hidden_size, hidden_size * 4)
        self.rgcn = RGCN(hidden_size, hidden_size, relation_num)
        # self.attn = Neighbor_Attn(input_size, hidden_size)

    def forward(self, x, h, c, g, adjs):
        '''

        :param x:   (node, emb)
            embedding of the node, news and initial node embedding
        :param h:   (node, hidden)
            hidden state from last layer
        :param c:   candidate from last layer
        :param g:   (hidden)
            hidden state of the global node
        :param adj:   (node, node)
            if use RGCN, there should be multiple gcns, each one for a relation
        :return:
        '''
        hn = self.rgcn(h, adjs)
        gates = self.Wh(h) + self.U(x) + self.Wn(hn) + torch.unsqueeze(self.V(g), 0)
        i, f, o, u = torch.split(gates, self.hidden_size, dim=-1)
        new_c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(u)
        new_h = torch.sigmoid(o) * torch.tanh(new_c)
        return new_h, new_c


class GLSTMCell(nn.Module):
    def __init__(self, hidden_size, attn_pooling):
        super(GLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.w = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U = nn.Linear(hidden_size, hidden_size * 2)
        self.u = nn.Linear(hidden_size, hidden_size)
        self.attn_pooling = attn_pooling

    def forward(self, g, c_g, h, c):
        ''' assume dim=1 is word'''
        # this can use attentive pooling
        # h_avg = torch.mean(h, 1)
        h_avg = self.attn_pooling(h)
        f, o = torch.split(torch.sigmoid(self.W(g) + self.U(h_avg)), self.hidden_size, dim=-1)
        f_w = torch.sigmoid(torch.unsqueeze(self.w(g), -2) + self.u(h))
        f_w = F.softmax(f_w, -2)
        new_c = f * c_g + torch.sum(c * f_w, -2)
        new_g = o * torch.tanh(new_c)
        return new_g, new_c


class GLSTM(nn.Module):
    def __init__(self, config, word_vocab):
        super(GLSTM, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.emb_size = config.emb_size
        self.word_vocab = word_vocab
        if word_vocab.emb is not None:
            self.word_emb = nn.Embedding.from_pretrained(torch.from_numpy(word_vocab.emb), freeze=config.freeze_emb)
        else:
            self.word_emb = nn.Embedding(word_vocab.voc_size, config.emb_size)
        self.node_emb = nn.Embedding(config.node_num, config.hidden_size)
        if config.encoder == 'cnn':
            self.text_encoder = models.CNN_Encoder(config.filter_size, config.hidden_size)
        elif config.encoder == 'rnn':
            self.text_encoder = models.RNN_Encoder(config.hidden_size, config.emb_size, config.dropout)
        self.feature_weight = nn.Linear(config.feature_size, config.hidden_size)
        self.feature_lstm = models.LSTM(config.hidden_size, config.hidden_size, config.dropout, bidirec=False)
        self.feature_combine = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.attn_pooling = Attentive_Pooling(config.hidden_size)
        assert config.graph_encoder in {'lstm', 'gru', 'highway', 'rgcn'}
        if config.graph_encoder == 'lstm':
            self.s_cell = SLSTMCell(config.hidden_size, config.hidden_size, config.relation_num)
            self.g_cell = GLSTMCell(config.hidden_size, self.attn_pooling)
        elif config.graph_encoder == 'gru':
            self.s_cell = SGRUCell(config.hidden_size, config.hidden_size, config.relation_num)
            self.g_cell = GGRUCell(config.hidden_size, self.attn_pooling)
        elif config.graph_encoder == 'highway':
            self.highway = highway_RGCN(config.hidden_size, config.hidden_size, config.relation_num)
        elif config.graph_encoder == 'rgcn':
            self.rgcn = RGCN(config.hidden_size, config.hidden_size, config.relation_num)
        self.w_out = nn.Linear(config.hidden_size, config.label_size)
        self.num_layers = config.num_layers
        self.dropout = torch.nn.Dropout(config.dropout)

    def encode(self, node_vector, adj):
        """

        :param span_nodes: (time, node)
        :param text_nodes: (node)
        :param node_text:   (node, seq)
        :param text_length:
        :param text_mask:  (node, seq)
        :param node_feature:   (time, node, feature_size)
        :param adj:
        :return:
        """
        last_h_layer = node_vector
        last_g_layer = None
        if self.config.graph_encoder == 'lstm':
            last_c_layer = node_vector
            last_g_layer = self.attn_pooling(last_h_layer)
            last_c_g_layer = self.attn_pooling(last_c_layer)
            for l in range(self.num_layers):
                # x, h, c, g, adj
                last_h_layer, last_c_layer = self.s_cell(node_vector, last_h_layer, last_c_layer, last_g_layer, adj)
                # g, c_g, h, c
                last_g_layer, last_c_g_layer = self.g_cell(last_g_layer, last_c_g_layer, last_h_layer, last_c_layer)
                last_h_layer = self.dropout(last_h_layer)
                last_g_layer = self.dropout(last_g_layer)
        elif self.config.graph_encoder == 'gru':
            last_g_layer = self.attn_pooling(last_h_layer)
            for l in range(self.num_layers):
                last_h_layer = self.s_cell(node_vector, last_h_layer, last_g_layer, adj)
                last_g_layer = self.g_cell(last_g_layer, last_h_layer)
        elif self.config.graph_encoder == 'highway':
            for l in range(self.num_layers):
                last_h_layer = self.highway(last_h_layer, adj)
        elif self.config.graph_encoder == 'rgcn':
            for l in range(self.num_layers):
                last_h_layer = self.rgcn(last_h_layer, adj)

        return last_h_layer, last_g_layer

    def forward(self, span_nodes, bert_vec, node_text, word_mask, sent_mask, node_feature, adj):
        '''
        :param batch:
            nodes: (time, graph_node)
            node_text: (time, node, seq)
            adjs: (time, node, node)
        :param use_cuda:
        :return:    (node, label)
        '''
        text_length = word_mask.sum(-1).int()
        assert text_length.max() <= node_text.size(-1) and text_length.min() > 0, (text_length, node_text.size())
        node_size, seq_size = node_text.size()
        node_emb = self.node_emb(span_nodes)
        word_emb = self.dropout(self.word_emb(node_text))
        text_vector = self.text_encoder(word_emb.view([-1, seq_size, self.emb_size]),
                                        word_mask.view(-1, seq_size), lengths=text_length.view(-1), query=node_emb)
        node_vector = self.feature_combine(torch.cat([text_vector, node_emb], -1))
        output, g_output = self.encode(node_vector, adj)
        if g_output is not None:
            return self.w_out(output), self.w_out(g_output)
        else:
            return self.w_out(output), self.w_out(torch.mean(output, 0))
