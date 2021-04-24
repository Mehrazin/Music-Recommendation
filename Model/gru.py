import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import dataloader
import os
import pickle
import numpy as np
import pandas as pd
import random
def create_embeding(config):
    """
    Creates an embedding matrix for the songs based on the lyrics embeddings
    """
    vocab = config.vocab
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
    return embedding, emb_dim, num_vocab

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        embedding, emb_dim, _ = create_embeding(config)
        self.embedding = embedding
        self.hidden_size = emb_dim
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)

    def forward(self, input):
        # input shape: (seq_len, N)
        embedded = self.embedding(input)
        embedded = embedded.float()
        # embedded shape: (seq_len, N, emb_len)
        output, hidden = self.gru(embedded)
        #output (seq_len, N, emb_len)
        # hidden (1,N,emb_len)
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        embedding, emb_dim, num_vocab = create_embeding(config)
        self.embedding = embedding
        self.hidden_size = emb_dim
        self.output_size = num_vocab
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.fc =  nn.Linear(self.hidden_size, self.output_size)
    def forward(self, input, hidden):
        #input: (N)
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        embedded = embedded.float()
        # (1, N, emb_size)
        output, hidden = self.gru(embedded, hidden)
        prediction = self.fc(output)
        # (1, N, num_vocab)
        prediction = prediction.squeeze(0)
        # (N, num_vocab)
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, config, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, config, users, source, target):
        target_len = target[0].shape[0]
        vocab_size = self.decoder.output_size
        batch_size = target[0].shape[1]

        outputs = torch.zeros(target_len, batch_size, vocab_size).to(config.device)

        _, hidden = self.encoder(source[0])
        x = target[0][0]
        for t in range(1, target_len):
            output, hidden = self.decoder(x, hidden)

            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[0][t] if random.random() < config.teacher_force_ratio else best_guess

        return outputs
