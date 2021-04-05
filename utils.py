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

        # Data preprocessing configurations
        self.max_valid_seq_len = 500
        self.max_session_len = 30
        self.min_session_len = 2
        self.min_session_per_user = 5
        self.time_interval = 3600
        self.load_raw_data = False
        self.keep_lang = 'en'
        self.col_lang_detect = ['artist_name']
        self.col_lang_name = ['artist_lang']
        self.test_mode = False

# class LanguageIdentification:
#
#     def __init__(self):
#         pretrained_lang_model = "lid.176.bin"
#         self.model = fasttext.load_model(pretrained_lang_model)
#
#     def predict_lang(self, text):
#         predictions = self.model.predict(text) # returns top 2 matching languages
#         return predictions
