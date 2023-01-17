from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize

from text2props.model import Text2PropsModel
from text2props.modules.estimators_from_text import (
    FeatureEngAndRegressionPipeline,
    FeatureEngAndRegressionEstimatorFromText,
)
from text2props.modules.feature_engineering import FeatureEngineeringModule
from text2props.modules.feature_engineering.components import (
    ReadabilityFeaturesComponent,
    IRFeaturesComponent,
    LinguisticFeaturesComponent,
    Word2VecFeaturesComponent,
)
from text2props.modules.feature_engineering.utils import vectorizer_text_preprocessor as prepr
from text2props.modules.latent_traits_calibration import KnownParametersCalibrator
from text2props.modules.regression import RegressionModule
from text2props.modules.regression.components import SklearnRegressionComponent

from ._data_collection import get_difficulty_range, get_dict_latent_traits
from ..configs import *


def get_text2props_model_by_config_and_dataset(config, dataset, random_seed):
    difficulty_range = get_difficulty_range(dataset)
    dict_latent_traits = get_dict_latent_traits(dataset)
    model = get_text2props_model_by_config(config, difficulty_range, dict_latent_traits, random_seed)
    return model


def get_text2props_model_by_config(config, difficulty_range, dict_latent_traits, seed):
    model = Text2PropsModel(
        latent_traits_calibrator=KnownParametersCalibrator(dict_latent_traits),
        estimator_from_text=FeatureEngAndRegressionEstimatorFromText(
            {
                DIFFICULTY: FeatureEngAndRegressionPipeline(
                    FeatureEngineeringModule(get_text2props_feat_eng_components_from_config(config, seed), normalize_method=normalize),
                    RegressionModule(get_text2props_regression_components_from_config(config, difficulty_range, seed)))
            }
        )
    )
    return model


def text2props_randomized_cv_train(model: Text2PropsModel, dict_params, df_train, random_seed, n_iter=20, n_jobs=-1):
    model.calibrate_latent_traits(None)
    scores = model.randomized_cv_train(dict_params, df_train=df_train, n_iter=n_iter, cv=5, n_jobs=n_jobs, random_state=random_seed)  # TODO pass params
    return scores


def get_predictions_text2props(model, df_train, df_test):
    y_pred_train = model.predict(df_train)[DIFFICULTY]
    y_pred_test = model.predict(df_test)[DIFFICULTY]
    return y_pred_train, y_pred_test


def get_predictions_r2de(model, x_train, x_test):
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    return y_pred_train, y_pred_test


def get_text2props_feat_eng_components_from_config(config, seed):

    feature_engineering_config = config.split('__')[0]

    if feature_engineering_config == LING:
        return [LinguisticFeaturesComponent(version=2)]
    if feature_engineering_config == READ:
        return [ReadabilityFeaturesComponent(use_smog=False, version=2)]

    if feature_engineering_config == W2V_Q_ONLY:
        return [Word2VecFeaturesComponent(size=100, seed=seed)]
    if feature_engineering_config == W2V_Q_ALL:
        return [Word2VecFeaturesComponent(size=100, concatenate_correct=True, concatenate_wrong=True, seed=seed)]
    if feature_engineering_config == W2V_Q_CORRECT:
        return [Word2VecFeaturesComponent(size=100, concatenate_correct=True, seed=seed)]

    if feature_engineering_config == LING_AND_READ:
        return [LinguisticFeaturesComponent(version=2),
                ReadabilityFeaturesComponent(use_smog=False, version=2)]

    if feature_engineering_config == W2V_Q_ONLY_AND_LING:
        return [Word2VecFeaturesComponent(size=100, seed=seed),
                LinguisticFeaturesComponent(version=2)]
    if feature_engineering_config == W2V_Q_ALL_AND_LING:
        return [Word2VecFeaturesComponent(size=100, concatenate_correct=True, concatenate_wrong=True, seed=seed),
                LinguisticFeaturesComponent(version=2)]
    if feature_engineering_config == W2V_Q_CORRECT_AND_LING:
        return [Word2VecFeaturesComponent(size=100, concatenate_correct=True, seed=seed),
                LinguisticFeaturesComponent(version=2)]

    if feature_engineering_config == LING_AND_READ_AND_R2DE_Q_ONLY:
        return [
            LinguisticFeaturesComponent(version=2),
            IRFeaturesComponent(
                TfidfVectorizer(stop_words='english', preprocessor=prepr, min_df=0.05, max_df=0.95, max_features=1000),
                concatenate_correct=False,
                concatenate_wrong=False,
            ),
        ]
    if feature_engineering_config == LING_AND_READ_AND_R2DE_Q_CORRECT:
        return [
            LinguisticFeaturesComponent(version=2),
            IRFeaturesComponent(
                TfidfVectorizer(stop_words='english', preprocessor=prepr, min_df=0.05, max_df=0.95, max_features=1000),
                concatenate_correct=True,
                concatenate_wrong=False,
            ),
        ]
    if feature_engineering_config == LING_AND_READ_AND_R2DE_Q_ALL:
        return [
            LinguisticFeaturesComponent(version=2),
            IRFeaturesComponent(
                TfidfVectorizer(stop_words='english', preprocessor=prepr, min_df=0.05, max_df=0.95, max_features=1000),
                concatenate_correct=True,
                concatenate_wrong=True,
            ),
        ]

    if feature_engineering_config == W2V_Q_ONLY_AND_LING_AND_R2DE_Q_ONLY:
        return [
            Word2VecFeaturesComponent(size=100, concatenate_correct=False, concatenate_wrong=False, seed=seed),
            LinguisticFeaturesComponent(version=2),
            IRFeaturesComponent(
                TfidfVectorizer(stop_words='english', preprocessor=prepr, min_df=0.05, max_df=0.95, max_features=1000),
                concatenate_correct=False,
                concatenate_wrong=False,
            ),
        ]
    if feature_engineering_config == W2V_Q_ONLY_AND_LING_AND_R2DE_Q_CORRECT:
        return [
            Word2VecFeaturesComponent(size=100, concatenate_correct=False, concatenate_wrong=False, seed=seed),
            LinguisticFeaturesComponent(version=2),
            IRFeaturesComponent(
                TfidfVectorizer(stop_words='english', preprocessor=prepr, min_df=0.05, max_df=0.95, max_features=1000),
                concatenate_correct=True,
                concatenate_wrong=False,
            ),
        ]
    if feature_engineering_config == W2V_Q_ONLY_AND_LING_AND_R2DE_Q_ALL:
        return [
            Word2VecFeaturesComponent(size=100, concatenate_correct=False, concatenate_wrong=False, seed=seed),
            LinguisticFeaturesComponent(version=2),
            IRFeaturesComponent(
                TfidfVectorizer(stop_words='english', preprocessor=prepr, min_df=0.05, max_df=0.95, max_features=1000),
                concatenate_correct=True,
                concatenate_wrong=True,
            ),
        ]

    if feature_engineering_config == W2V_Q_ALL_AND_LING_AND_R2DE_Q_ONLY:
        return [
            Word2VecFeaturesComponent(size=100, concatenate_correct=True, concatenate_wrong=True, seed=seed),
            LinguisticFeaturesComponent(version=2),
            IRFeaturesComponent(
                TfidfVectorizer(stop_words='english', preprocessor=prepr, min_df=0.05, max_df=0.95, max_features=1000),
                concatenate_correct=False,
                concatenate_wrong=False,
            ),
        ]
    if feature_engineering_config == W2V_Q_ALL_AND_LING_AND_R2DE_Q_CORRECT:
        return [
            Word2VecFeaturesComponent(size=100, concatenate_correct=True, concatenate_wrong=True, seed=seed),
            LinguisticFeaturesComponent(version=2),
            IRFeaturesComponent(
                TfidfVectorizer(stop_words='english', preprocessor=prepr, min_df=0.05, max_df=0.95, max_features=1000),
                concatenate_correct=True,
                concatenate_wrong=False,
            ),
        ]
    if feature_engineering_config == W2V_Q_ALL_AND_LING_AND_R2DE_Q_ALL:
        return [
            Word2VecFeaturesComponent(size=100, concatenate_correct=True, concatenate_wrong=True, seed=seed),
            LinguisticFeaturesComponent(version=2),
            IRFeaturesComponent(
                TfidfVectorizer(stop_words='english', preprocessor=prepr, min_df=0.05, max_df=0.95, max_features=1000),
                concatenate_correct=True,
                concatenate_wrong=True,
            ),
        ]

    if feature_engineering_config == W2V_Q_CORRECT_AND_LING_AND_R2DE_Q_ONLY:
        return [
            Word2VecFeaturesComponent(size=100, concatenate_correct=True, concatenate_wrong=False, seed=seed),
            LinguisticFeaturesComponent(version=2),
            IRFeaturesComponent(
                TfidfVectorizer(stop_words='english', preprocessor=prepr, min_df=0.05, max_df=0.95, max_features=1000),
                concatenate_correct=False,
                concatenate_wrong=False,
            ),
        ]
    if feature_engineering_config == W2V_Q_CORRECT_AND_LING_AND_R2DE_Q_CORRECT:
        return [
            Word2VecFeaturesComponent(size=100, concatenate_correct=True, concatenate_wrong=False, seed=seed),
            LinguisticFeaturesComponent(version=2),
            IRFeaturesComponent(
                TfidfVectorizer(stop_words='english', preprocessor=prepr, min_df=0.05, max_df=0.95, max_features=1000),
                concatenate_correct=True,
                concatenate_wrong=False,
            ),
        ]
    if feature_engineering_config == W2V_Q_CORRECT_AND_LING_AND_R2DE_Q_ALL:
        return [
            Word2VecFeaturesComponent(size=100, concatenate_correct=True, concatenate_wrong=False, seed=seed),
            LinguisticFeaturesComponent(version=2),
            IRFeaturesComponent(
                TfidfVectorizer(stop_words='english', preprocessor=prepr, min_df=0.05, max_df=0.95, max_features=1000),
                concatenate_correct=True,
                concatenate_wrong=True,
            ),
        ]


def get_text2props_regression_components_from_config(config, difficulty_range, seed):
    regression_config = config.split('__')[1]
    if regression_config == LR:
        return [SklearnRegressionComponent(LinearRegression(), latent_trait_range=difficulty_range)]
    if regression_config == RF:
        return [SklearnRegressionComponent(RandomForestRegressor(random_state=seed), latent_trait_range=difficulty_range)]