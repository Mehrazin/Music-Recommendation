"""
This module contains files for preproccesing the data and defining sessions and also train-test split

"""
from utils import Config
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
    def __init__(self,config):
        self.data = load_data(config)
        self.users = self.get_val_list('user_id')
        self.artists = self.get_val_list('artist_name')
        self.tracks = self.get_val_list('track_name')


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


















if __name__ == '__main__':
    config = Config()
    data = load_data(config)
    # data = lang_detect(data, config)
    data = create_session(data, config)
