"""
This file contains class and modules for configuration and setup.

"""
import os

class Config:
    def __init__(self):
        # Define dataset path
        Dataset_dir = os.path.join(os.getcwd(), 'Dataset')
        Original_lastfm_dir = os.path.join(Dataset_dir, 'lastfm-dataset-1K')
        Original_user_track_dir = os.path.join(Original_lastfm_dir, 'userid-timestamp-artid-artname-traid-traname.tsv')
        Original_user_profile_dir = os.path.join(Original_lastfm_dir, 'userid-profile.tsv')
        Processed_dataset_dir = os.path.join(Dataset_dir, 'Final_dataset.csv')
        Test_dataset_dir = os.path.join(Dataset_dir, 'Test_sample.csv')

        self.max_valid_seq_len = 500
        self.max_session_len = 30
        self.min_session_len = 2
        self.min_session_per_user = 5
        self.time_interval = 3600
        self.load_raw_data = False
        self.keep_lang = 'en'
        self.col_lang_detect = ['artist_name']
        self.col_lang_name = ['artist_lang']
        self.test_mode = True

class LanguageIdentification:

    def __init__(self):
        pretrained_lang_model = "lid.176.bin"
        self.model = fasttext.load_model(pretrained_lang_model)

    def predict_lang(self, text):
        predictions = self.model.predict(text) # returns top 2 matching languages
        return predictions
