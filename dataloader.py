from utils import Config
import pandas as pd
import pickle
import os
import numpy as np

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
