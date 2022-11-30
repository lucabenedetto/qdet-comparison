import pandas as pd
import pickle
from text2props.constants import DIFFICULTY, Q_ID


def get_difficulty_range(dataset_name):
    return -1, 3


def get_output_dir(dataset_name):  # todo by dataset
    output_dir = './output/race_pp'
    return output_dir


def get_dfs(dataset_name):  # todo by dataset
    df_train = pd.read_csv('./data/processed/t2p_race_pp_train.csv')
    df_test = pd.read_csv('./data/processed/t2p_race_pp_test.csv')
    return df_train, df_test


def get_lts(dataset_name):  # todo by dataset
    dict_latent_traits = pickle.load(open('./data/processed/t2p_race_pp_difficulty_dict.p', "rb"))
    return dict_latent_traits


def get_true_labels(df_train, df_test, dataset_name):
    dict_latent_traits = get_lts(dataset_name)
    y_true_test = [dict_latent_traits[DIFFICULTY][x] for x in df_test[Q_ID].values]
    y_true_train = [dict_latent_traits[DIFFICULTY][x] for x in df_train[Q_ID].values]
    return y_true_train, y_true_test
