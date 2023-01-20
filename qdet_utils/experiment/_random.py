from typing import Optional
import os
import pandas as pd
import pickle
import random

from qdet_utils.experiment import BaseExperiment
from qdet_utils.constants import Q_ID, OUTPUT_DIR, DATA_DIR


class RandomExperiment(BaseExperiment):
    def __init__(
            self,
            dataset_name: str,
            data_dir: str = DATA_DIR,
            output_root_dir: str = OUTPUT_DIR,
            random_seed: Optional[int] = None,
    ):
        super().__init__(dataset_name, data_dir, output_root_dir, random_seed)
        self.true_difficulties = None
        self.min_diff = None
        self.max_diff = None

    def get_dataset(self):
        # TODO I should redo this in order not to use the t2p dataframe
        self.df_train = pd.read_csv(os.path.join(self.data_dir, f't2p_{self.dataset_name}_train.csv'), dtype={Q_ID: str})
        self.df_test = pd.read_csv(os.path.join(self.data_dir, f't2p_{self.dataset_name}_test.csv'), dtype={Q_ID: str})
        self.y_true_train = pickle.load(open(os.path.join(self.data_dir, f'y_true_train_{self.dataset_name}.p'), 'rb'))
        self.y_true_dev = pickle.load(open(os.path.join(self.data_dir, f'y_true_dev_{self.dataset_name}.p'), 'rb'))
        self.y_true_test = pickle.load(open(os.path.join(self.data_dir, f'y_true_test_{self.dataset_name}.p'), 'rb'))

    def init_model(self, pretrained_model: Optional = None, model_name: str = 'model', *args, **kwargs):
        self.model_name = model_name

    def train(self, *args, **kwargs):
        if self.discrete_regression:
            self.true_difficulties = list(set(self.y_true_train))
        else:
            self.min_diff = min(self.y_true_train)
            self.max_diff = max(self.y_true_train)

    def predict(self):
        if self.discrete_regression:
            self.y_pred_train = [random.choice(self.true_difficulties) for _ in range(len(self.df_train))]
            self.y_pred_test = [random.choice(self.true_difficulties) for _ in range(len(self.df_test))]
        else:
            self.y_pred_train = [random.random() * (self.max_diff - self.min_diff) + self.min_diff for _ in range(len(self.df_train))]
            self.y_pred_test = [random.random() * (self.max_diff - self.min_diff) + self.min_diff for _ in range(len(self.df_test))]
        pickle.dump(self.y_pred_test, open(os.path.join(self.output_dir, 'predictions_test_' + self.model_name + '.p'), 'wb'))
        pickle.dump(self.y_pred_train, open(os.path.join(self.output_dir, 'predictions_train_' + self.model_name + '.p'), 'wb'))
