import os
import pandas as pd
import pickle
from text2props.constants import DIFFICULTY, Q_ID
from src.constants import OUTPUT_DIR, RACE_PP, ARC, AM, DATA_DIR


def get_difficulty_range(dataset):
    if dataset == RACE_PP:
        return -1, 3
    else:
        raise NotImplementedError


def get_output_dir(dataset):
    return os.path.join(OUTPUT_DIR, dataset)


def get_dfs(dataset):
    df_train = pd.read_csv(os.path.join(DATA_DIR, f't2p_{dataset}_train.csv'))
    df_test = pd.read_csv(os.path.join(DATA_DIR, f't2p_{dataset}_test.csv'))
    return df_train, df_test


def get_lts(dataset):
    return pickle.load(open(os.path.join(DATA_DIR, f't2p_{dataset}_difficulty_dict.p'), "rb"))


def get_true_labels(df_train, df_test, dataset):
    dict_latent_traits = get_lts(dataset)
    y_true_test = [dict_latent_traits[DIFFICULTY][x] for x in df_test[Q_ID].values]
    y_true_train = [dict_latent_traits[DIFFICULTY][x] for x in df_train[Q_ID].values]
    return y_true_train, y_true_test
