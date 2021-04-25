import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import dataloader
import os
import pickle
import numpy as np
import pandas as pd
import random
import torch.nn.functional as F
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

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.encoder_hidden_dim = config.emb_dim
        self.decoder_hidden_dim = config.emb_dim
        # The input dimension will the the concatenation of
        # encoder_hidden_dim (hidden) and  decoder_hidden_dim(encoder_outputs)
        self.attn_hidden_vector = nn.Linear(self.encoder_hidden_dim + self.decoder_hidden_dim, self.decoder_hidden_dim)

        # We need source len number of values for n batch as the dimension
        # of the attention weights. The attn_hidden_vector will have the
        # dimension of [source len, batch size, decoder hidden dim]
        # If we set the output dim of this Linear layer to 1 then the
        # effective output dimension will be [source len, batch size]
        self.attn_scoring_fn = nn.Linear(self.decoder_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [1, batch size, decoder hidden dim]
        src_len = encoder_outputs.shape[0]

        # We need to calculate the attn_hidden for each source words.
        # Instead of repeating this using a loop, we can duplicate
        # hidden src_len number of times and perform the operations.
        hidden = hidden.repeat(src_len, 1, 1)

        # Calculate Attention Hidden values
        attn_hidden = torch.tanh(self.attn_hidden_vector(torch.cat((hidden, encoder_outputs), dim=2)))

        # Calculate the Scoring function. Remove 3rd dimension.
        attn_scoring_vector = self.attn_scoring_fn(attn_hidden).squeeze(2)

        # The attn_scoring_vector has dimension of [source len, batch size]
        # Since we need to calculate the softmax per record in the batch
        # we will switch the dimension to [batch size,source len]
        attn_scoring_vector = attn_scoring_vector.permute(1, 0)

        # Softmax function for normalizing the weights to
        # probability distribution
        return F.softmax(attn_scoring_vector, dim=1)

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        embedding, emb_dim, num_vocab = create_embeding(config)
        self.embedding = embedding
        self.hidden_size = emb_dim
        self.output_size = num_vocab
        self.attention = Attention(config)
        self.gru = nn.GRU(self.hidden_size*2, self.hidden_size)
        self.fc =  nn.Linear(self.hidden_size, self.output_size)
    def forward(self, input, hidden, encoder_outputs):
        #input: (N)
        input = input.unsqueeze(0)

        embedded = self.embedding(input)
        embedded = embedded.float()
        # Calculate the attention weights
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)

        # We need to perform the batch wise dot product.
        # Hence need to shift the batch dimension to the front.
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # Use PyTorch's bmm function to calculate the weight W.
        W = torch.bmm(a, encoder_outputs)

        # Revert the batch dimension.
        W = W.permute(1, 0, 2)

        # concatenate the previous output with W
        gru_input = torch.cat((embedded, W), dim=2)

        output, hidden = self.gru(gru_input, hidden)
        # (N, num_vocab)
        return output, hidden

class User_combine(nn.Module):
    def __init__(self, config):
        super(User_combine, self).__init__()
        self.num_vocab = len(list(config.vocab.idx2emb.keys())) + 2
        self.emb_dim = config.emb_dim
        self.num_user = len(list(config.vocab.idx2user.keys())) + 1
        self.embedding = nn.Embedding(self.num_user, self.emb_dim)
        self.fc_u = nn.Linear(self.emb_dim, self.num_vocab, bias = False)
        self.fc_h = nn.Linear(self.emb_dim, self.num_vocab, bias = False)
    def forward(self, user, decoder_output):
        user_embed = self.embedding(user)
        decoder_output = decoder_output.squeeze(0)
        u = self.fc_u(user_embed)
        h = self.fc_h(decoder_output)
        prediction = u + h
        return prediction

class Seq2Seq(nn.Module):
    def __init__(self, config, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.user_combine = User_combine(config)

    def forward(self, config, users, source, target):
        target_len = target[0].shape[0]
        vocab_size = self.decoder.output_size
        batch_size = target[0].shape[1]

        outputs = torch.zeros(target_len, batch_size, vocab_size).to(config.device)

        enc_output, hidden = self.encoder(source[0])
        x = target[0][0]
        for t in range(1, target_len):
            output, hidden = self.decoder(x, hidden, enc_output)
            output = self.user_combine(users, output)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[0][t] if random.random() < config.teacher_force_ratio else best_guess

        return outputs
