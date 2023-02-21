from typing import Optional, Dict, List
import os
import pandas as pd
import pickle
from sklearn.model_selection import RandomizedSearchCV

from r2de.encoding import get_encoded_texts
from r2de.model import get_model

from qdet_utils.experiment import BaseExperiment
from qdet_utils.constants import OUTPUT_DIR, DATA_DIR


class R2deExperiment(BaseExperiment):
    def __init__(
            self,
            dataset_name: str,
            data_dir: str = DATA_DIR,
            output_root_dir: str = OUTPUT_DIR,
            random_seed: Optional[int] = None,
    ):
        super().__init__(dataset_name, data_dir, output_root_dir, random_seed)
        self.encoding_idx = None
        self.x_train = None
        self.x_test = None

    def get_dataset(self):
        pass
        self.df_train = pd.read_csv(os.path.join(self.data_dir, f'r2de_{self.dataset_name}_train.csv'))
        self.df_test = pd.read_csv(os.path.join(self.data_dir, f'r2de_{self.dataset_name}_test.csv'))
        self.y_true_train = pickle.load(open(os.path.join(self.data_dir, f'y_true_train_{self.dataset_name}.p'), 'rb'))
        self.y_true_dev = pickle.load(open(os.path.join(self.data_dir, f'y_true_dev_{self.dataset_name}.p'), 'rb'))
        self.y_true_test = pickle.load(open(os.path.join(self.data_dir, f'y_true_test_{self.dataset_name}.p'), 'rb'))

    def init_model(
            self,
            pretrained_model: Optional = None,
            model_name: str = 'model',
            encoding_idx: int = None,
            *args, **kwargs
    ):
        if pretrained_model:
            # This is to use if the model is already defined and I don't have to train it
            raise NotImplementedError()
        else:
            self.model_name = model_name
            self.encoding_idx = encoding_idx
            self.model = get_model()

    # train with randomized CV and save model
    def train(
            self,
            dict_params: Dict[str, List[Dict[str, List[float]]]] = None,
            n_iter: int = 10,
            n_jobs: int = None,
            cv: int = 5,
            *args, **kwargs,
    ):
        self.x_train, self.x_test = get_encoded_texts(self.encoding_idx, self.df_train, self.df_test)
        random_search = RandomizedSearchCV(self.model, dict_params, n_iter=n_iter, cv=cv, n_jobs=n_jobs, random_state=self.random_seed)
        random_search.fit(self.x_train, self.y_true_train)
        self.model = random_search.best_estimator_
        # best_scores = random_search.best_score_
        pickle.dump(self.model, open(os.path.join(self.output_dir, f'model_r2de_encoding_{self.encoding_idx}.p'), 'wb'))

    def predict(self, save_predictions: bool = True):
        self.y_pred_train = self.model.predict(self.x_train)
        self.y_pred_test = self.model.predict(self.x_test)
        if save_predictions:
            pickle.dump(self.y_pred_test, open(os.path.join(self.output_dir, f'predictions_test_r2de_encoding_{self.encoding_idx}.p'), 'wb'))
            pickle.dump(self.y_pred_train, open(os.path.join(self.output_dir, f'predictions_train_r2de_encoding_{self.encoding_idx}.p'), 'wb'))
