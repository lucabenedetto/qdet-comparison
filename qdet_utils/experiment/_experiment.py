from typing import Optional
import os

from qdet_utils.evaluation import evaluate_model
from qdet_utils.constants import (
    RACE_PP, ARC, ARC_BALANCED, AM, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K,
    DATA_DIR, OUTPUT_DIR,
)


class BaseExperiment:
    def __init__(
            self,
            dataset_name: str,
            data_dir: str = DATA_DIR,
            output_root_dir: str = OUTPUT_DIR,
            random_seed: Optional[int] = None,
    ):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.output_data_dir = output_root_dir
        self.random_seed = random_seed

        self.model = None
        self.model_name = None  # Used for logging and in the filenames for saving the results.
        self.output_dir = os.path.join(self.output_data_dir, self.dataset_name, 'seed_' + str(self.random_seed))

        self.df_train = None
        self.df_test = None
        self.my_mapper = self.get_mapper(self.dataset_name)
        self.discrete_regression = self.get_discrete_regression(self.dataset_name)
        self.y_true_train = None
        self.y_true_test = None
        self.y_true_dev = None
        self.y_pred_train = None
        self.y_pred_test = None
        self.y_pred_dev = None

    def get_dataset(self):
        raise NotImplementedError()

    def init_model(self, pretrained_model, model_name, feature_eng_and_regression_pipeline):
        raise NotImplementedError()

    def train(self, dict_params, n_iter, n_jobs, cv):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()

    def evaluate(self):
        converted_y_pred_test = [self.my_mapper(x) for x in self.y_pred_test]
        converted_y_pred_train = [self.my_mapper(x) for x in self.y_pred_train]
        evaluate_model(root_string=self.model_name, y_pred_test=converted_y_pred_test, y_pred_train=converted_y_pred_train,
                       y_true_test=self.y_true_test, y_true_train=self.y_true_train,
                       output_dir=self.output_dir, discrete_regression=self.discrete_regression)

    # # # # # # # # # # # # # # # #

    @staticmethod
    def get_discrete_regression(dataset_name):
        return dataset_name in {RACE_PP, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K, ARC, ARC_BALANCED}

    def get_mapper(self, dataset_name: str):
        if dataset_name in {RACE_PP, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K}:
            return self.mapper_race
        if dataset_name in {ARC, ARC_BALANCED}:
            return self.mapper_arc
        if dataset_name == AM:
            return self.mapper_am
        else:
            raise NotImplementedError

    # TODO these mappers should be probably moved somewhere else

    @staticmethod
    def mapper_race(x):
        if x <= 0.5:
            return 0
        elif x < 1.5:
            return 1
        else:
            return 2

    @staticmethod
    def mapper_am(x):
        return x

    @staticmethod
    def mapper_arc(x):
        if x < 3.5:
            return 3
        elif x >= 8.5:
            return 9
        else:
            return round(x)

    # # # # # # # # # # # # #

    def get_difficulty_range(self, dataset):
        if dataset in {RACE_PP, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K}:
            return -1, 3
        if dataset in {ARC, ARC_BALANCED}:
            return 3, 9
        if dataset in {AM}:
            return -5, +5
        else:
            raise NotImplementedError
