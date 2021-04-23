from utils import Config
import pandas as pd
import pickle
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Vocab:
    def __init__(self, config):
        self.user2idx = {}
        self.idx2user = {}
        self.item2idx = {}
        self.idx2item = {}
        self.idx2emb = {}
        self.item2emb = {}

        if config.gen_user_idx :
            self.get_user_idx(config)
        if config.gen_item_idx :
            self.get_item_idx(config)
        if config.gen_item_emb:
            self.get_item_emb(config)
        if config.gen_emb_idx :
            self.get_emb_idx(config)
        if config.save_vocab:
            self.save(config)

    def get_user_idx(self, config):
        """
        Prepares user2id and id2user dictionaries
        """
        path = os.path.join(config.Dataset_dir, 'users.csv')
        temp = pd.read_csv(path)
        temp.drop_duplicates(inplace = True)
        temp.sort_values(['user_id'], ascending=True, inplace=True)
        temp.reset_index(inplace = True, drop = True)
        temp.reset_index(inplace = True)
        dic = {int(int(row['index']) + 1) : row['user_id'] for _,row in temp.iterrows()}
        assert (type(0) in set([type(k) for k in dic.keys()])) and len(set([type(k) for k in dic.keys()])) == 1
        self.idx2user = dic
        dic = {v: k for k,v in self.idx2user.items()}
        assert (type('str') in set([type(k) for k in dic.keys()])) and len(set([type(k) for k in dic.keys()])) == 1
        self.user2idx = dic
        del temp, dic

    def get_item_idx(self, config):
        """
        Prepares item2id and id2item dictionaries
        """
        path = os.path.join(config.Dataset_dir, 'items.csv')
        temp = pd.read_csv(path)
        temp.drop_duplicates(inplace = True)
        temp.sort_values(['artist_name', 'track_name'], ascending=True, inplace=True)
        temp.reset_index(inplace = True, drop = True)
        temp.reset_index(inplace = True)
        dic = {int(int(row['index']) + 2) : (row['artist_name'], row['track_name']) for _,row in temp.iterrows()}
        assert (type(0) in set([type(k) for k in dic.keys()])) and len(set([type(k) for k in dic.keys()])) == 1
        self.idx2item = dic
        dic = {v: k for k,v in dic.items()}
        assert (type(('1','2')) in set([type(k) for k in dic.keys()])) and len(set([type(k) for k in dic.keys()])) == 1
        self.item2idx = dic
        del temp, dic

    def get_item_emb(self, config):
        path = path = os.path.join(config.Dataset_dir, 'doc2vec_vectors.pkl')
        with open(path, 'rb') as f:
            vec = pickle.load(f)
        path = os.path.join(config.Dataset_dir, 'items.csv')
        items = pd.read_csv(path)
        items.reset_index(inplace = True, drop = True)
        items.reset_index(inplace = True)
        dic = {(row['artist_name'], row['track_name']) : vec[int(row['index'])] for _,row in items.iterrows()}
        self.item2emb = dic
        assert (type(('1','2')) in set([type(k) for k in dic.keys()])) and len(set([type(k) for k in dic.keys()])) == 1

    def get_emb_idx(self, config):
        dic = {i : self.item2emb[self.idx2item[i]] for i in list(self.idx2item.keys())}
        self.idx2emb = dic
        assert (type(0) in set([type(k) for k in dic.keys()])) and len(set([type(k) for k in dic.keys()])) == 1

    def save(self, config):
        path = os.path.join(config.exp_dir, 'vocab.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self, f)


class Music_data(Dataset):
    def __init__(self, config, dtype):
        super(Music_data, self).__init__()
        self.data = []
        self.dtype = dtype
        self.pad_index = config.pad_index
        self.sos_index = config.sos_index
        if dtype == 'train':
            self.path = config.train_dir
        if dtype == 'valid':
            self.path = config.valid_dir
        if dtype == 'test':
            self.path = config.test_dir
        assert os.path.exists(self.path)

        vocab_path = os.path.join(config.Dataset_dir, 'vocab.pkl')
        assert os.path.exists(vocab_path)
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        self.vocab = vocab
        self.load_data(config)

    def load_data(self, config):
        """
        Loads the data from the disk and put to self.data
        """
        data = pd.read_csv(self.path)
        def str_to_int(st):
            st = st.rstrip().split(',')
            l = [int(t) for t in st]
            return l
        df = [(row['user_id'], str_to_int(row['input']), str_to_int(row['output'])) for _,row in data.iterrows()]
        self.data = df
        self.size = len(self.data)

    def __len__(self):
        """
        Retuens the size of the dataset
        """
        return self.size

    def __getitem__(self, idx):
        """
        Returns an item in index = idx from the dataset
        """
        return self.data[idx]

    def create_batch(self, sequences):
        """
        Creates a batch from the given list of equations.
        """
        lengths = torch.LongTensor([len(s) + 2 for s in sequences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.pad_index)
        assert lengths.min().item() > 2
        sent[0] = self.sos_index
        for i, s in enumerate(sequences):
            sent[1:lengths[i] - 1, i].copy_(s)
            sent[lengths[i] - 1, i] = self.sos_index
        return sent, lengths

    def collate_fn(self, elements):
        """
        Collate samples into a batch.
        """
        u, i, o = zip(*elements)
        u = [torch.LongTensor([self.vocab.user2idx[user]]) for user in u]
        i = [torch.LongTensor([token for token in seq]) for seq in i]
        o = [torch.LongTensor([token for token in seq]) for seq in o]
        i, in_len = self.create_batch(i)
        o, out_len = self.create_batch(o)
        return u, (i, in_len), (o, out_len)

def create_data_loader(config, dtype, shuffle = False):
    """
    Returns an iterable to iterate over the data. dtype can be one of the followings:
    'train', 'valid', 'test'
    """
    assert dtype in ['train', 'valid', 'test']
    dataset = Music_data(config, dtype)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn
    )
