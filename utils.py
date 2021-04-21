"""
This file contains class and modules for configuration and setup.

"""
import os
# import fasttext

class Config:
    def __init__(self):
        # Define dataset path
        self.Dataset_dir = os.path.join(os.getcwd(), 'Dataset')
        self.Original_lastfm_dir = os.path.join(self.Dataset_dir, 'lastfm-dataset-1K')
        self.Original_user_track_dir = os.path.join(self.Original_lastfm_dir, 'userid-timestamp-artid-artname-traid-traname.tsv')
        self.Original_user_profile_dir = os.path.join(self.Original_lastfm_dir, 'userid-profile.tsv')
        self.Processed_dataset_dir = os.path.join(self.Dataset_dir, 'Final_dataset.csv')
        self.Test_dataset_dir = os.path.join(self.Dataset_dir, 'Test_sample_2.csv')
        # Experiment path
        self.dump_path = os.path.join(os.getcwd(), 'Dumped')
        if not os.path.exists(self.dump_path):
            os.mkdir(self.dump_path)
        self.exp_id = self.get_exp_id()
        self.exp_dir = os.path.join(self.dump_path, str(self.exp_id))
        if not os.path.exists(self.exp_dir):
            os.mkdir(self.exp_dir)
        # Final files Path
        self.vocab_dir = os.path.join(self.exp_dir, 'vocab.plk')
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
        self.test_mode = True
        self.sanity_check = True
        self.pad_index = 0
        self.sos_index = 1
        self.dump_vocab = True
        # Vocabulary configurations
        self.gen_user_idx = True
        self.gen_item_idx = True

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

# class LanguageIdentification:
#
#     def __init__(self):
#         pretrained_lang_model = "lid.176.bin"
#         self.model = fasttext.load_model(pretrained_lang_model)
#
#     def predict_lang(self, text):
#         predictions = self.model.predict(text) # returns top 2 matching languages
#         return predictions
