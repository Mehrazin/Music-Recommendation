"""
This module contains files for preproccesing the data and defining sessions and also train-test split

"""
from utils import Config, bool_flag
import pandas as pd
from tqdm import tqdm
import numpy as np
from dataloader import Vocab
import pickle
import os
import time
from multiprocessing import  Pool
import multiprocessing as mp
from lyricsgenius import Genius
import argparse



def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=bool_flag, default=False)
    parser.add_argument('--load_vocab', type=bool_flag, default=True)
    return parser.parse_args()

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

# def lang_detect(data, config):
#     """
#     Detects the language of elements of a particular list of columns and add it to the dataset
#     """
#     pred = LanguageIdentification()
#     columns = config.col_lang_detect
#     out_name = config.col_lang_name
#     for index, col in tqdm(enumerate(columns)):
#         langs = []
#         for item in tqdm(data[col]):
#             item = item.replace("\n", " ")
#             lang = pred.predict_lang(item)[0]
#             langs.append(str(lang)[11:13])
#         data[out_name[index]] = langs
#     return data

class Data_handler():
    """
    This class helps to handle the dataset easier by providing useful atributes and methods.
    """
    def __init__(self,config, **kwargs):
        if 'data' in kwargs.keys():
            self.data = kwargs['data']
        else:
            self.data = load_data(config)
        if 'pass_col' in kwargs.keys():
            self.pass_col = kwargs['pass_col']
        else :
            self.pass_col = []
        self.info = {}
        self.update()

    def update(self):
        for col in list(self.data.columns):
            if col in self.pass_col:
                pass
            else:
                self.info[col] = self.get_val_list(col)



    def get_val_list(self, col_name):
        assert col_name in list(self.data.columns)
        values = list(self.data[col_name])
        values = list(dict.fromkeys(values))
        return values

    def get_masked_data(self, users, fracs, method = 'from_the_top'):
        """
        This method outputs a dataset that has a fraction of the original dataset's users.
        Both users and fractions are givven as input to the method. This methid is very useful
        in unit-testing process.
        """
        assert len(users) == len(fracs)
        if method == 'from_the_top':
            sub_data = pd.DataFrame(columns = self.data.columns)
            for i, user in enumerate(users) :
                temp_df = self.data[self.data.user_id == user]
                new_len = int(float(fracs[i]*len(temp_df)))
                temp_df = temp_df[:new_len]
                sub_data = pd.concat([sub_data, temp_df])
            sub_data.reset_index(inplace = True, drop = True)
        return sub_data






def clean_data(data, config):
    if 'rm_non_en' in config.clean_mode:
        # Remove non-English values
        if config.keep_lang:
            for col_name in config.col_lang_name:
                data = data[data[col_name] == config.keep_lang]
    if 'rm_small_sess' in config.clean_mode:
        # Remove session that are too small
        temp_df = pd.DataFrame(data.Session.value_counts())
        temp_df.reset_index(inplace = True)
        temp_df.columns = ['Session', 'len']
        if config.sanity_check:
            sc = pd.DataFrame(temp_df[temp_df.len < config.min_session_len])
            not_val_sess = list(sc['Session'])
        temp_df = temp_df[temp_df.len >= config.min_session_len]
        temp_df = temp_df[['Session']]
        data = pd.merge(data, temp_df, how = 'inner', on = ["Session"])
        data.sort_values(['user_id', 'timestamp'], ascending=True, inplace=True)
        data.reset_index(inplace = True, drop = True)
        if config.sanity_check:
            curr_sess = list(data.Session)
            curr_sess = list(dict.fromkeys(curr_sess))
            for sess in not_val_sess:
                assert sess not in curr_sess
    if 'rm_small_sub_session' in config.clean_mode:
        # Remove sub-sessions with small length
        temp_df = pd.DataFrame(data.groupby(['Session', 'sub_session'])['timestamp'].count())
        temp_df.reset_index(inplace = True)
        temp_df.columns = ['Session', 'sub_session', 'len']
        temp_df = temp_df[temp_df.len >= config.min_session_len]
        temp_df = temp_df[['Session', 'sub_session']]
        data = pd.merge(data, temp_df, how = 'inner', on = ["Session", 'sub_session'])
        data.sort_values(['user_id', 'timestamp'], ascending=True, inplace=True)
        data.reset_index(inplace = True, drop = True)

    if 'rm_small_users' in config.clean_mode:
        # Remove users with small number of sessions
        temp_df = pd.DataFrame(data.groupby('user_id')['Session'].nunique())
        temp_df.reset_index(inplace = True)
        temp_df.columns = ['user_id', 'len']
        if config.sanity_check:
            sc = pd.DataFrame(temp_df[temp_df.len < config.min_session_per_user])
            not_val_users = list(sc['user_id'])
        temp_df = temp_df[temp_df.len >= config.min_session_per_user]
        temp_df = temp_df[['user_id']]
        data = pd.merge(data, temp_df, how = 'inner', on = ["user_id"])
        data.sort_values(['user_id', 'timestamp'], ascending=True, inplace=True)
        data.reset_index(inplace = True, drop = True)
        if config.sanity_check:
            curr_users = list(data.user_id)
            curr_users = list(dict.fromkeys(curr_sess))
            for user in not_val_users:
                assert user not in curr_users
    return data

def cut_sessions(data, config):
    """
    This function divides each session whose length is greater than config.max_session_len to sub_sessions whose length equals to max_session_len.
    """
    data['sub_session'] = ''
    temp_df = pd.DataFrame(data.Session.value_counts())
    temp_df.reset_index(inplace = True)
    temp_df.columns = ['Session', 'len']
    sess = list(temp_df.Session)
    l = list(temp_df.len)
    dic = dict(zip(sess, l))
    curr_sess = list(data.Session)
    curr_sess = list(dict.fromkeys(curr_sess))
    sub = np.array([0]).reshape((1,-1))
    for se in curr_sess :
        if dic[se] > config.max_session_len :
            i = int(dic[se]/config.max_session_len)
            r = dic[se] - i*config.max_session_len
            x = np.arange(1, i+1).reshape((1,-1))
            l = np.repeat(x, config.max_session_len, axis=1)
            if r != 0:
                a = np.arange(i+1, i+2).reshape((1,-1))
                a = np.repeat(a, r, axis=1)
                l = np.append(l, a, axis = 1)
        else:
            x = np.arange(1, 2).reshape((1,-1))
            l = np.repeat(x, dic[se], axis=1)
        sub = np.append(sub,l, axis = 1)
    sub = sub.astype(int)
    sub = sub[0][1:].reshape((1,-1))
    data['sub_session'] = np.squeeze(sub)
    return data

def create_session(data, config):
    """
    Creates sessions for each user. Also, it devides each session to subsessions based on config.max_session_len
    """
    data.sort_values(['user_id', 'timestamp'], ascending=True, inplace=True)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.reset_index(inplace = True, drop = True)
    data['Session'] = ''
    new_data = pd.DataFrame(columns = data.columns)
    users = list(data.user_id)
    users = list(dict.fromkeys(users))
    for index, user in enumerate(users):
        sess_idx = 1
        sub_data = pd.DataFrame(data[data['user_id'] == user])
        prev_time = 0
        sessions = []
        for i, time in enumerate(sub_data['timestamp']):
            if prev_time == 0:
                prev_time = time
            elif (time - prev_time).total_seconds() < config.time_interval :
                prev_time = time
            else:
                sess_idx +=1
                prev_time = time
            sessions.append('U' + str(index) + 'S' + str(sess_idx))
            # sub_data.loc[i, 'Session'] = 'U' + str(index) + 'S' + str(sess_idx)
        sub_data['Session'] = sessions
        new_data = pd.concat([new_data, sub_data])
        del sub_data
        # data.loc[data.user_id == user, 'Session'] = sub_data.Session
    new_data.dropna(inplace = True)
    new_data.reset_index(inplace = True, drop = True)
    assert len(new_data) == len(data)
    return new_data

def prepare_train_data(data, config):
    # data contains user_id, timestamp, artist_name, track_name, session, sub_session
    if config.dump_vocab:
        file_path = config.vocab_dir
        with open(file_path, 'rb') as f:
            vocab = pickle.load(f)
        print('Vocabulary loaded')
    else:
        vocab = get_vocab(data, config)
    data['idx'] = data.apply(lambda row: vocab.item2idx[(row['artist_name'], row['track_name'])], axis = 1)
    print('Index Assigned')
    df = Data_handler(config, data = data)
    # df.sessions = df.get_val_list('Session')
    new_df = pd.DataFrame(columns = ['input', 'output', 'in_len', 'user_id'])
    # for i, session in enumerate(df.sessions):
    for i, session in enumerate(df.info['Session']):
        data = df.data[df.data.Session == session]
        sess_df = Data_handler(config, data = data)
        # sess_df.sub_sessions = sess_df.get_val_list('sub_session')
        if i%10 == 0 :
            # print(f'Session: {session} for user: {sess_df.users[0]} is under process')
            print(f'Session: {session} for user: {sess_df.info["user_id"][0]} is under process')
        # for sub in sess_df.sub_sessions:
        for sub in sess_df.info['sub_session']:
            temp_df = pd.DataFrame()
            temp_sub = sess_df.data[sess_df.data['sub_session'] == sub]
            In = []
            In_len = []
            Out = []
            for i_len in range(config.min_in_sess_len, len(temp_sub)):
                In_len.append(i_len)
                i = list(temp_sub[:i_len].idx)
                o = list(temp_sub[i_len:].idx)
                i = [str(x) for x in i]
                i = ','.join(i)
                o = [str(x) for x in o]
                o = ','.join(o)
                In.append(i)
                Out.append(o)
            temp_df['input'] = In
            temp_df['output'] = Out
            temp_df['in_len'] = In_len
            assert len(sess_df.info['user_id']) > 0
            temp_df['user_id'] = sess_df.info['user_id'][0]
            new_df = pd.concat([new_df, temp_df])
    new_df = new_df[['user_id', 'in_len', 'input', 'output']]
    return new_df




def get_vocab(data, config):
    """
    Creates and outputs a Vocabulary object for the current data
    """
    # data contains user_id, timestamp, artist_name, track_name, session, sub_session
    vocab = Vocab(data,config)
    if config.dump_vocab:
        file_path = config.vocab_dir
        with open(file_path, 'wb') as f:
            pickle.dump(vocab, f)
    else:
        return vocab

def train_valid_test_split(data, config):
    """
    Splits train, validation, and test datasets. Also, removes items from test and val set that have not seen in the training set
    """
    # data contains user_id, timestamp, artist_name, track_name, session, sub_session
    df = Data_handler(config, data = data)
    train_idx = 0
    val_idx = 0
    train_session = []
    val_session = []
    test_session = []
    for user in df.info['user_id']:
        data = df.data[df.data.user_id == user]
        user_df = Data_handler(config, data = data)
        # user_df.sessions = user_df.get_val_list('Session')
        if len(user_df.info['Session']) <= 20:
            train_idx = len(user_df.info['Session']) - 2
            val_idx = train_idx + 1
        else:
            train_idx = int(len(user_df.info['Session']) * 0.9)
            val_idx = train_idx + int(len(user_df.info['Session']) * 0.05)
        train_session.extend(user_df.info['Session'][:train_idx])
        val_session.extend(user_df.info['Session'][train_idx:val_idx])
        test_session.extend(user_df.info['Session'][val_idx:])
        types = {'train': train_session, 'val' : val_session, 'test': test_session}
    output = {}
    for task in types.keys():
        temp = pd.DataFrame()
        temp['Session'] = types[task]
        output[task] = pd.DataFrame(pd.merge(df.data, temp, how = 'inner', on = ["Session"]))
    train = output['train']
    seen_items = pd.DataFrame(train[['artist_name', 'track_name']])
    seen_items.drop_duplicates(subset = ['artist_name', 'track_name'], inplace = True, ignore_index = True)
    for key in output.keys():
        temp = pd.merge(output[key], seen_items, how = 'inner', on = ['artist_name', 'track_name'])
        temp.sort_values(['user_id', 'timestamp'], ascending=True, inplace=True)
        temp.reset_index(inplace = True, drop = True)
        output[key] = temp
    return output

def save_data(config):
    """
    The input data has all the attributes: user_id, timestamp, session, sub_session, lyrics
    """
    # Keep only user_id, timestamp, artist_name, track_name, session, sub_session
    path = os.path.join(config.Dataset_dir, 'Dataset.csv')
    data = pd.read_csv(path)
    output = train_valid_test_split(data, config)
    print('Train-test split was done')
    processed_output = {}
    for task in (output.keys()):
        processed_output[task] = prepare_train_data(output[task], config)
    print('Data prepared to be saved')
    for task in (output.keys()):
        name_raw = task + '_raw.csv'
        path_raw = os.path.join(config.exp_dir, name_raw)
        name_clean = task + '_clean.csv'
        path_clean = os.path.join(config.exp_dir, name_clean)
        output[task].to_csv(path_raw, index = False)
        processed_output[task].to_csv(path_clean, index = False)
    print('Dataset saved')



# Lyrics part
class song_not_found(Exception):
    pass

def init_genius(token):
    # get the genius instanse
    genius = Genius(token)
    # Turn off status messages
    genius.verbose = False
    # Remove section headers (e.g. [Chorus]) from lyrics when searching
    genius.remove_section_headers = True
    # Include hits thought to be non-songs (e.g. track lists)
    genius.skip_non_songs = False
    # Exclude songs with these words in their title
    genius.excluded_terms = ["(Remix)", "(Live)"]
    return genius

def create_subdf(df, start, stop, drop_labels):
    sub_df = df[start:stop].copy()
    sub_df.drop(labels = drop_labels, axis = 1, inplace = True)
    return sub_df

def fetch_lyrics(artist_name, track_name, genius):
    try:
        song = genius.search_song(title=track_name, artist=artist_name, get_full_info=True)
        if not song :
            return np.nan
        lyrics = song.lyrics
        lyrics = lyrics.lower()
        if lyrics.islower():
            return song.lyrics
        else:
            return np.nan
    except :
        return np.nan
def add_lyrics(df):
    token = 'K_biyrE1lISJN24FOY3d1oQBP34PA0rYo4WTxyPriLSNvfH49v7QFHOu3BM2Q0RU'
    genius = init_genius(token)
    df['lyrics'] = df.apply(lambda row : fetch_lyrics(row['artist_name'], row['track_name'], genius), axis = 1)
    return df

def parallelize_dataframe(df, func, n_cores):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def lyrics(data, config):
    tracks = pd.DataFrame(data[['artist_name','track_name']])
    tracks.drop_duplicates(['artist_name','track_name'], inplace = True)
    tracks.reset_index(inplace = True, drop = True)
    final_df = pd.DataFrame(columns = ['artist_name', 'track_name', 'lyrics'])
    num_itr = int(len(tracks)/1000) + 1
    for i in tqdm(range(num_itr)):
        sub_tracks = create_subdf(tracks, i*1000, (i+1)*1000, [])
        tic = time.perf_counter()
        sub_tracks = parallelize_dataframe(sub_tracks, add_lyrics, n_cores = mp.cpu_count())
        toc = time.perf_counter()
        print(f'Performance: {toc - tic:0.8f} seconds')
        final_df = pd.concat([final_df, sub_tracks])
        final_df.to_csv(config.data_w_lyrics_dir, index = False)
        del sub_tracks
        final_df = pd.read_csv(config.data_w_lyrics_dir)
    final_df = pd.read_csv(config.data_w_lyrics_dir)
    final_df = final_df.dropna(inplace = True)
    final_df['lyrics'] = final_df['lyrics'].apply(lambda x: x.replace("\n", " "))
    final_df.to_csv(config.processed_data_w_lyrics_dir, index = False)

def merge_val_item (config):
    """
    Based on the items that we have embeddings for, builds necessary datasets
    """
    config.test_mode = False
    raw_df = Data_handler(config)
    path = os.path.join(config.Test_dir, 'df_doc2vec.csv')
    items = pd.read_csv(path)
    items.drop(columns = ['Unnamed: 0'], inplace = True)
    items.drop(columns = ['embedding'], inplace = True)
    new_df = pd.merge(raw_df.data, items, how = 'inner', on = ['artist_name', 'track_name'])
    users = pd.DataFrame(new_df[['user_id']])
    users.drop_duplicates(subset = ['user_id'], inplace = True, ignore_index = True)
    path = os.path.join(config.exp_dir, 'items.csv')
    items.to_csv(path, index = False)
    path = os.path.join(config.exp_dir, 'users.csv')
    users.to_csv(path, index = False)
    config.clean_mode = ['rm_small_sess', 'rm_small_users']
    assert config.clean_mode == ['rm_small_sess', 'rm_small_users']
    new_df = clean_data(new_df, config)
    config.clean_mode = ['rm_small_sub_session']
    new_df = clean_data(new_df, config)
    new_df.sort_values(['user_id', 'timestamp'], ascending=True, inplace=True)
    path = os.path.join(config.exp_dir, 'Dataset.csv')
    new_df.to_csv(path, index = False)

if __name__ == '__main__':
    arg = arg_parse()
    config = Config(arg)
    path = os.path.join(config.exp_dir, 'config.pkl')
    print(arg.test_mode)
    print(config.test_mode)
    # df = Data_handler(config)
    # lyrics(df.data, config)
    # with open(path, 'wb') as f:
    #     pickle.dump(config, f)
    # df.data = create_session(df.data, config)
    # df.data = clean_data(df.data, config)
    # df.data = cut_sessions(df.data, config)
    # config.clean_mode = ['rm_small_sub_session']
    # df.data = clean_data(df.data, config)
    # get_vocab(df.data, config)
    # new_df = prepare_train_data(df.data, config)
