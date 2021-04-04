"""
This module contains files for preproccesing the data and defining sessions and also train-test split

"""
from utils import Config, LanguageIdentification
import pandas as pd
from tqdm import tqdm

def load_data(config):
    """
    Load the original User-Track dataset
    """
    if config.load_raw_data :
        filepath = config.Original_user_track_dir
        user_track_data = pd.read_csv(
            filepath, sep='\t', header=None,
            names=[
                'user_id', 'timestamp', 'artist_id', 'artist_name', 'track_id', 'track_name'
            ],
            skiprows=[
                2120260-1, 2446318-1, 11141081-1,
                11152099-1, 11152402-1, 11882087-1,
                12902539-1, 12935044-1, 17589539-1
            ]
        )
        user_track_data["timestamp"] = pd.to_datetime(user_track_data.timestamp)
        user_track_data.sort_values(['user_id', 'timestamp'], ascending=True, inplace=True)
        user_track_data.dropna(inplace = True)
        user_track_data.reset_index(inplace = True, drop = True)
        print(f'Number of Records: {len(user_track_data):,}\nUnique Users: {user_track_data.user_id.nunique()}\nUnique Artist:{user_track_data.artist_id.nunique():,}')
    elif config.test_mode :
        filepath = config.Test_dataset_dir
        user_track_data = pd.read_csv(filepath)
    else :
        filepath = config.Processed_dataset_dir
        user_track_data = pd.read_csv(filepath)
    return user_track_data

def lang_detect(data, config):
    """
    Detects the language of elements of a particular list of columns and add it to the dataset
    """
    pred = LanguageIdentification()
    columns = config.col_lang_detect
    out_name = config.col_lang_name
    for index, col in tqdm(enumerate(columns)):
        langs = []
        for item in tqdm(data[col]):
            item = item.replace("\n", " ")
            lang = pred.predict_lang(item)[0]
            langs.append(str(lang)[11:13])
        data[out_name[index]] = langs
    return data

def clean_data(data, config):
    if config.keep_lang:
        for col_name in config.col_lang_name:
            data = data[data[col_name] == config.keep_lang]
    return data















if __name__ == '__main__':
    config = Config()
    data = load_data(config)
    data = lang_detect(data, config)
