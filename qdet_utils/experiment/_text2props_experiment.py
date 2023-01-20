from typing import Optional, Dict, List
import os
import pandas as pd
import pickle

from text2props.constants import DIFFICULTY
from text2props.model import Text2PropsModel
from text2props.modules.estimators_from_text import (
    FeatureEngAndRegressionPipeline,
    FeatureEngAndRegressionEstimatorFromText,
)
from text2props.modules.latent_traits_calibration import KnownParametersCalibrator

from qdet_utils.experiment import BaseExperiment
from qdet_utils.constants import Q_ID, OUTPUT_DIR, DATA_DIR


class Text2propsExperiment(BaseExperiment):
    def __init__(
            self,
            dataset_name: str,
            data_dir: str = DATA_DIR,
            output_root_dir: str = OUTPUT_DIR,
            random_seed: Optional[int] = None,
    ):
        super().__init__(dataset_name, data_dir, output_root_dir, random_seed)
        self.dict_latent_traits = None

    def get_dataset(self):
        self.df_train = pd.read_csv(os.path.join(self.data_dir, f't2p_{self.dataset_name}_train.csv'), dtype={Q_ID: str})
        self.df_test = pd.read_csv(os.path.join(self.data_dir, f't2p_{self.dataset_name}_test.csv'), dtype={Q_ID: str})
        self.y_true_train = pickle.load(open(os.path.join(self.data_dir, f'y_true_train_{self.dataset_name}.p'), 'rb'))
        self.y_true_dev = pickle.load(open(os.path.join(self.data_dir, f'y_true_dev_{self.dataset_name}.p'), 'rb'))
        self.y_true_test = pickle.load(open(os.path.join(self.data_dir, f'y_true_test_{self.dataset_name}.p'), 'rb'))
        self.dict_latent_traits = pickle.load(open(os.path.join(self.data_dir, f't2p_{self.dataset_name}_difficulty_dict.p'), "rb"))

    def init_model(
            self,
            pretrained_model: Optional[Text2PropsModel] = None,
            model_name: str = 'model',
            feature_eng_and_regression_pipeline: Optional[FeatureEngAndRegressionPipeline] = None,  # TODO
            *args, **kwargs,
    ):
        if pretrained_model:
            # This is to use if the model is already defined and I don't have to train it
            raise NotImplementedError()
        else:
            # Here I init a "new" model which will have to be trained
            self.model_name = model_name
            self.model = Text2PropsModel(
                latent_traits_calibrator=KnownParametersCalibrator(self.dict_latent_traits),
                estimator_from_text=FeatureEngAndRegressionEstimatorFromText({DIFFICULTY: feature_eng_and_regression_pipeline})
            )

    # train with randomized CV and save model
    def train(
            self,
            dict_params: Dict[str, List[Dict[str, List[float]]]],
            n_iter: int,
            n_jobs: int,
            cv: int = 5,
            *args, **kwargs,
    ):
        self.model.calibrate_latent_traits(None)
        scores = self.model.randomized_cv_train(
            dict_params,
            df_train=self.df_train,
            n_iter=n_iter,
            cv=cv,
            n_jobs=n_jobs,
            random_state=self.random_seed,
        )
        pickle.dump(self.model, open(os.path.join(self.output_dir, 'model_' + self.model_name + '.p'), 'wb'))

    def predict(self):
        self.y_pred_train = self.model.predict(self.df_train)[DIFFICULTY]
        self.y_pred_test = self.model.predict(self.df_test)[DIFFICULTY]
        pickle.dump(self.y_pred_test, open(os.path.join(self.output_dir, 'predictions_test_' + self.model_name + '.p'), 'wb'))
        pickle.dump(self.y_pred_train, open(os.path.join(self.output_dir, 'predictions_train_' + self.model_name + '.p'), 'wb'))
