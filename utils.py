"""
This file contains class and modules for configuration and setup.

"""
import os
import argparse
import pickle
import torch
# import fasttext
FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")

class Config:
    def __init__(self, arg):
        # Define dataset path
        self.Dataset_dir = os.path.join(os.getcwd(), 'Dataset')
        self.Original_lastfm_dir = os.path.join(self.Dataset_dir, 'lastfm-dataset-1K')
        self.Original_user_track_dir = os.path.join(self.Original_lastfm_dir, 'userid-timestamp-artid-artname-traid-traname.tsv')
        self.Original_user_profile_dir = os.path.join(self.Original_lastfm_dir, 'userid-profile.tsv')
        self.Processed_dataset_dir = os.path.join(self.Dataset_dir, 'Final_df_w_sessions.csv')
        self.Test_dir = os.path.join(self.Dataset_dir, 'Test', '1k-item', 'Datasets')
        assert os.path.exists(self.Test_dir)
        if arg.test_mode:
            self.Dataset_dir = self.Test_dir

        self.train_dir = os.path.join(self.Dataset_dir, 'train_clean.csv')
        self.test_dir = os.path.join(self.Dataset_dir, 'test_clean.csv')
        self.valid_dir = os.path.join(self.Dataset_dir, 'val_clean.csv')

        # self.Test_dataset_dir = os.path.join(self.Test_dir, 'df_lyrics.csv')
        # Experiment path
        self.dump_path = os.path.join(os.getcwd(), 'Dumped')
        if not os.path.exists(self.dump_path):
            os.mkdir(self.dump_path)
        self.exp_id = self.get_exp_id()
        self.exp_dir = os.path.join(self.dump_path, str(self.exp_id))
        if not os.path.exists(self.exp_dir):
            os.mkdir(self.exp_dir)
        # Final files Path
        self.vocab_dir = os.path.join(self.Dataset_dir, 'vocab.pkl')
        self.data_w_lyrics_dir = os.path.join(self.exp_dir, 'added_lyrics.csv')
        self.processed_data_w_lyrics_dir = os.path.join(self.exp_dir, 'processed_lyrics.csv')
        # Data preprocessing configurations
        self.max_valid_seq_len = 500
        self.max_session_len = 20
        self.min_session_len = 2
        self.min_session_per_user = 5
        self.min_in_sess_len = 2
        self.share_in_out_item = 1
        self.max_in_sess_len = self.max_session_len - self.min_in_sess_len
        self.time_interval = 3600
        self.load_raw_data = False
        self.keep_lang = 'en'
        self.clean_mode = ['rm_small_sess', 'rm_small_users']
        self.col_lang_detect = ['artist_name']
        self.col_lang_name = ['artist_lang']
        self.test_mode = arg.test_mode
        self.sanity_check = True
        self.pad_index = 0
        self.sos_index = 1
        self.dump_vocab = True
        # Vocabulary configurations
        self.gen_user_idx = True
        self.gen_item_idx = True
        self.gen_item_emb = True
        self.gen_emb_idx = True
        self.save_vocab = True
        # Embedding configurations
        self.tfidf_vector_size = 500
        self.doc2vec_vector_size = 500
        self.doc2vec_epoochs = 50
        # Training configurations
        self.batch_size = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emb_freeze = True
        self.save()



    def get_exp_id(self):
        """
        Returns an integer as an experiment id.
        """
        dir_list = os.listdir(self.dump_path)
        if len(dir_list) == 0 :
            id = 1
        else :
            dir_list = [int(dir) for dir in dir_list]
            id = max(dir_list) + 1
        return id
    def save(self):
        path = os.path.join(self.exp_dir, 'config.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self, f)

# class LanguageIdentification:
#
#     def __init__(self):
#         pretrained_lang_model = "lid.176.bin"
#         self.model = fasttext.load_model(pretrained_lang_model)
#
#     def predict_lang(self, text):
#         predictions = self.model.predict(text) # returns top 2 matching languages
#         return predictions
