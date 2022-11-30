from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize  # w_normalized = normalize(w, norm='l1', axis=1)

from text2props.constants import DIFFICULTY
from text2props.model import Text2PropsModel
from text2props.modules.estimators_from_text import (
    FeatureEngAndRegressionPipeline,
    FeatureEngAndRegressionEstimatorFromText,
)
from text2props.modules.feature_engineering import FeatureEngineeringModule
from text2props.modules.feature_engineering.components import ReadabilityFeaturesComponent, IRFeaturesComponent
from text2props.modules.latent_traits_calibration import KnownParametersCalibrator
from text2props.modules.regression import RegressionModule
from text2props.modules.regression.components import SklearnRegressionComponent
from text2props.modules.feature_engineering.utils import vectorizer_text_preprocessor as preproc

from ._data_collection import get_difficulty_range, get_latent_traits
from ..configs import *


def get_model_by_config_and_dataset(config, dataset):
    difficulty_range = get_difficulty_range(dataset)
    dict_latent_traits = get_latent_traits(dataset)
    model = get_model_by_config(config, difficulty_range, dict_latent_traits)
    return model


def get_model_by_config(config, difficulty_range, dict_latent_traits):
    model = Text2PropsModel(
        latent_traits_calibrator=KnownParametersCalibrator(dict_latent_traits),
        estimator_from_text=FeatureEngAndRegressionEstimatorFromText(
            {
                DIFFICULTY: FeatureEngAndRegressionPipeline(
                    FeatureEngineeringModule(get_feat_eng_components_from_config(config), normalize_method=normalize),
                    RegressionModule(get_regression_components_from_config(config, difficulty_range)))
            }
        )
    )
    return model


def randomized_cv_train(model, df_train, random_seed):
    model.calibrate_latent_traits(None)
    dict_params = {DIFFICULTY: [{}]}  # todo make argument
    scores = model.randomized_cv_train(dict_params, df_train=df_train, n_iter=50, cv=5, n_jobs=-1, random_state=random_seed)  # TODO pass params
    return scores


def get_predictions(model, df_train, df_test):
    y_pred_train = model.predict(df_train)[DIFFICULTY]
    y_pred_test = model.predict(df_test)[DIFFICULTY]
    return y_pred_train, y_pred_test


def get_feat_eng_components_from_config(config):
    if config == READABILITY_AND_R2DE__LR:
        return [
            ReadabilityFeaturesComponent(),
            IRFeaturesComponent(
                TfidfVectorizer(stop_words='english', preprocessor=preproc, min_df=0.1, max_df=1.0, max_features=1394),
                # values above are from the R2DE CV training
                concatenate_correct=True,
                concatenate_wrong=True
            )
        ]


def get_regression_components_from_config(config, difficulty_range):
    if config == READABILITY_AND_R2DE__LR:
        return [SklearnRegressionComponent(LinearRegression(), latent_trait_range=difficulty_range)]
