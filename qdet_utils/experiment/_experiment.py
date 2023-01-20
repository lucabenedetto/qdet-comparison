from typing import Optional
import os

from qdet_utils.difficulty_mapping_methods import get_difficulty_mapper
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
        self.output_root_dir = output_root_dir
        self.random_seed = random_seed

        self.model = None
        self.model_name = None  # Used for logging and in the filenames for saving the results.
        if self.random_seed is not None:
            self.output_dir = os.path.join(self.output_root_dir, self.dataset_name, 'seed_' + str(self.random_seed))
        else:
            self.output_dir = os.path.join(self.output_root_dir, self.dataset_name)

        self.df_train = None
        self.df_test = None
        self.difficulty_mapper = get_difficulty_mapper(self.dataset_name)
        self.discrete_regression = self.get_discrete_regression(self.dataset_name)
        self.y_true_train = None
        self.y_true_test = None
        self.y_true_dev = None
        self.y_pred_train = None
        self.y_pred_test = None
        self.y_pred_dev = None

    def get_dataset(self):
        raise NotImplementedError()

    def init_model(self, pretrained_model: Optional = None, model_name: str = 'model', *args, **kwargs):
        raise NotImplementedError()

    def train(self, *args, **kwargs):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()

    def evaluate(self, compute_correlation: bool = True):
        converted_y_pred_test = [self.difficulty_mapper(x) for x in self.y_pred_test]
        converted_y_pred_train = [self.difficulty_mapper(x) for x in self.y_pred_train]
        evaluate_model(
            model_name=self.model_name,
            y_pred_test=converted_y_pred_test,
            y_pred_train=converted_y_pred_train,
            y_true_test=self.y_true_test,
            y_true_train=self.y_true_train,
            output_dir=self.output_dir,
            discrete_regression=self.discrete_regression,
            compute_correlation=compute_correlation,
        )

    @staticmethod
    def get_discrete_regression(dataset_name):
        return dataset_name in {RACE_PP, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K, ARC, ARC_BALANCED}

    def get_difficulty_range(self, dataset):
        if dataset in {RACE_PP, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K}:
            return -1, 3
        if dataset in {ARC, ARC_BALANCED}:
            return 3, 9
        if dataset in {AM}:
            return -5, +5
        else:
            raise NotImplementedError
