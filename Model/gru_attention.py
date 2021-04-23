import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import dataloader
import os
import pickle
import numpy as np
import pandas as pd

def create_embeding(config):
    vocab_path = os.path.join(config.Dataset_dir, 'vocab.pkl')
    assert os.path.exists(vocab_path)
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    num_vocab = len(list(vocab.idx2emb.keys())) + 2
    emb_dim = vocab.idx2emb[2].shape[0]
    assert type(emb_dim) == int
    emb_matrix = np.zeros((num_vocab, emb_dim))
    for i in range(num_vocab):
        if i == 0:
            pass
        elif i == 1:
            emb_matrix[i] = np.ones(emb_dim)
        else:
            emb_matrix[i] = vocab.idx2emb[i]
    weights = torch.from_numpy(emb_matrix)
    embedding = nn.Embedding.from_pretrained(weights, padding_idx = 0, freeze=config.emb_freeze)
    return embedding, emb_dim

class EncoderRNN(nn.Module):
    def __init__(self, config):
        super(EncoderRNN, self).__init__()
        embedding, emb_dim = create_embeding(config)
        self.embedding = embedding
        self.hidden_size = emb_dim
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)

    def forward(self, input):
        # input shape: (seq_len, N)
        embedded = self.embedding(input)
        # embedded shape: (seq_len, N, emb_len)
        output, hidden = self.gru(embedded)
        return output, hidden


class Decoder(nn.Module):
    pass

class Seq2Seq(nn.Module):
    pass
