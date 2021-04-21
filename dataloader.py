from utils import Config
import pandas as pd


class Vocab:
    def __init__(self, data, config):
        self.user2idx = {}
        self.idx2user = {}
        self.item2idx = {}
        self.idx2item = {}
        self.emb2idx = {}
        self.idx2emb = {}

        if config.gen_user_idx :
            self.get_user_idx(data)
        if config.gen_item_idx :
            self.get_item_idx(data)

    def get_user_idx(self, data):
        """
        Prepares user2id and id2user dictionaries
        """
        temp = pd.DataFrame(data.user_id)
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

    def get_item_idx(self, data):
        """
        Prepares item2id and id2item dictionaries
        """
        temp = pd.DataFrame(data[['artist_name', 'track_name']])
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
