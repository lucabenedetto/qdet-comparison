import os
import pickle
from text2props.constants import DIFFICULTY, Q_ID
from qdet_utils.constants import DATA_DIR


def get_true_labels(df_train, df_test, dataset):
    dict_latent_traits = pickle.load(open(os.path.join(DATA_DIR, f't2p_{dataset}_difficulty_dict.p'), "rb"))
    y_true_test = [dict_latent_traits[DIFFICULTY][x] for x in df_test[Q_ID].values]
    y_true_train = [dict_latent_traits[DIFFICULTY][x] for x in df_train[Q_ID].values]
    return y_true_train, y_true_test
