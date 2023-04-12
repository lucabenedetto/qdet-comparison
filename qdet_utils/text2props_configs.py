from typing import Dict, Any, Optional
from scipy.stats import randint
from text2props.constants import DIFFICULTY
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize

from text2props.modules.feature_engineering import FeatureEngineeringModule
from text2props.modules.feature_engineering.components import (
    ReadabilityFeaturesComponent,
    IRFeaturesComponent,
    LinguisticFeaturesComponent,
    Word2VecFeaturesComponent,
)
from text2props.modules.feature_engineering.utils import vectorizer_text_preprocessor
from text2props.modules.regression import RegressionModule
from text2props.modules.regression.components import SklearnRegressionComponent


def text2props_get_dict_params(config: str, param_distribution: Optional[Dict[str, Any]] = None):
    if param_distribution is not None:
        return param_distribution
    regressor_config = config.split('__')[1]
    if regressor_config == LR:
        return {DIFFICULTY: [{}]}
    if regressor_config == RF:
        return {DIFFICULTY: [{'regressor__n_estimators': randint(20, 200), 'regressor__max_depth': randint(2, 50)}]}
    raise NotImplementedError


def text2props_get_config(feature_engineering_config, regression_config):
    assert feature_engineering_config in FEATURE_ENGINEERING_CONFIGS
    assert regression_config in REGRESSION_CONFIGS
    return feature_engineering_config + '__' + regression_config


# FEAT ENG. PARTS
LING = 'ling'
READ = 'read'
W2V_Q_ONLY = 'w2v_q_only'
W2V_Q_ALL = 'w2v_q_all'
W2V_Q_CORRECT = 'w2v_q_correct'
# below the hybrid approaches
LING_AND_READ = 'ling_and_read'
W2V_Q_ONLY_AND_LING = 'w2v_q_only_and_ling'
W2V_Q_ALL_AND_LING = 'w2v_q_all_and_ling'
W2V_Q_CORRECT_AND_LING = 'w2v_q_correct_and_ling'
LING_AND_READ_AND_R2DE_Q_ONLY = 'ling_and_read_and_r2de_q_only'
LING_AND_READ_AND_R2DE_Q_CORRECT = 'ling_and_read_and_r2de_q_correct'
LING_AND_READ_AND_R2DE_Q_ALL = 'ling_and_read_and_r2de_q_all_q_all'
W2V_Q_ONLY_AND_LING_AND_R2DE_Q_ONLY = 'w2v_q_only_and_ling_and_r2de_q_only'
W2V_Q_ONLY_AND_LING_AND_R2DE_Q_CORRECT = 'w2v_q_only_and_ling_and_r2de_q_correct'
W2V_Q_ONLY_AND_LING_AND_R2DE_Q_ALL = 'w2v_q_only_and_ling_and_r2de_q_all_q_all'
W2V_Q_ALL_AND_LING_AND_R2DE_Q_ONLY = 'w2v_q_all_and_ling_and_r2de_q_only'
W2V_Q_ALL_AND_LING_AND_R2DE_Q_CORRECT = 'w2v_q_all_and_ling_and_r2de_q_correct'
W2V_Q_ALL_AND_LING_AND_R2DE_Q_ALL = 'w2v_q_all_and_ling_and_r2de_q_all_q_all'
W2V_Q_CORRECT_AND_LING_AND_R2DE_Q_ONLY = 'w2v_q_correct_and_ling_and_r2de_q_only'
W2V_Q_CORRECT_AND_LING_AND_R2DE_Q_CORRECT = 'w2v_q_correct_and_ling_and_r2de_q_correct'
W2V_Q_CORRECT_AND_LING_AND_R2DE_Q_ALL = 'w2v_q_correct_and_ling_and_r2de_q_all'
FEATURE_ENGINEERING_CONFIGS = {
    LING,
    READ,
    W2V_Q_ONLY, W2V_Q_ALL, W2V_Q_CORRECT,
    LING_AND_READ,
    LING_AND_READ_AND_R2DE_Q_ONLY, LING_AND_READ_AND_R2DE_Q_CORRECT, LING_AND_READ_AND_R2DE_Q_ALL,
    W2V_Q_ONLY_AND_LING, W2V_Q_ALL_AND_LING, W2V_Q_CORRECT_AND_LING,
    W2V_Q_ONLY_AND_LING_AND_R2DE_Q_ONLY, W2V_Q_ONLY_AND_LING_AND_R2DE_Q_CORRECT, W2V_Q_ONLY_AND_LING_AND_R2DE_Q_ALL,
    W2V_Q_ALL_AND_LING_AND_R2DE_Q_ONLY, W2V_Q_ALL_AND_LING_AND_R2DE_Q_CORRECT, W2V_Q_ALL_AND_LING_AND_R2DE_Q_ALL,
    W2V_Q_CORRECT_AND_LING_AND_R2DE_Q_ONLY,
    W2V_Q_CORRECT_AND_LING_AND_R2DE_Q_CORRECT,
    W2V_Q_CORRECT_AND_LING_AND_R2DE_Q_ALL,
}
# below not implemented hybrid approaches
# LING_AND_R2DE = 'ling_and_r2de'
# READ_AND_R2DE = 'read_and_r2de'

# Regression elements
LR = 'LR'
RF = 'RF'
REGRESSION_CONFIGS = {LR, RF}

DEFAULT_W2V_SIZE = 100
DEFAULT_TFIDF_MAX_FEATS = 1000
DEFAULT_TFIDF_MIN_DF = 1000
DEFAULT_TFIDF_MAX_DF = 1000


def text2props_get_feature_engineering_module_from_config(config, seed) -> FeatureEngineeringModule:
    feature_engineering_config = config.split('__')[0]
    # All the possible components
    linguistic_features = LinguisticFeaturesComponent(version=2)
    readability_features = ReadabilityFeaturesComponent(use_smog=False, version=2)
    w2v_q_only = Word2VecFeaturesComponent(
        size=DEFAULT_W2V_SIZE, concatenate_correct=False, concatenate_wrong=False, seed=seed)
    w2v_q_all = Word2VecFeaturesComponent(
        size=DEFAULT_W2V_SIZE, concatenate_correct=True, concatenate_wrong=True, seed=seed)
    w2v_q_correct = Word2VecFeaturesComponent(
        size=DEFAULT_W2V_SIZE, concatenate_correct=True, concatenate_wrong=False, seed=seed)
    tfidf_q_only = IRFeaturesComponent(
        TfidfVectorizer(
            stop_words='english', preprocessor=vectorizer_text_preprocessor,
            min_df=DEFAULT_TFIDF_MIN_DF, max_df=DEFAULT_TFIDF_MAX_DF, max_features=DEFAULT_TFIDF_MAX_FEATS),
        concatenate_correct=False,
        concatenate_wrong=False,
    )
    tfidf_q_correct = IRFeaturesComponent(
        TfidfVectorizer(
            stop_words='english', preprocessor=vectorizer_text_preprocessor,
            min_df=DEFAULT_TFIDF_MIN_DF, max_df=DEFAULT_TFIDF_MAX_DF, max_features=DEFAULT_TFIDF_MAX_FEATS),
        concatenate_correct=True,
        concatenate_wrong=False,
    )
    tfidf_q_all = IRFeaturesComponent(
        TfidfVectorizer(
            stop_words='english', preprocessor=vectorizer_text_preprocessor,
            min_df=DEFAULT_TFIDF_MIN_DF, max_df=DEFAULT_TFIDF_MAX_DF, max_features=DEFAULT_TFIDF_MAX_FEATS),
        concatenate_correct=True,
        concatenate_wrong=True,
    )
    # the combinations
    dict_feat_eng_components = {
        LING: [linguistic_features],
        READ: [readability_features],
        W2V_Q_ONLY: [w2v_q_only],
        W2V_Q_ALL: [w2v_q_all],
        W2V_Q_CORRECT: [w2v_q_correct],
        LING_AND_READ: [linguistic_features, readability_features],
        W2V_Q_ONLY_AND_LING: [w2v_q_only, linguistic_features],
        W2V_Q_ALL_AND_LING: [w2v_q_all, linguistic_features],
        W2V_Q_CORRECT_AND_LING: [w2v_q_correct, linguistic_features],
        LING_AND_READ_AND_R2DE_Q_ONLY: [linguistic_features, tfidf_q_only],
        LING_AND_READ_AND_R2DE_Q_CORRECT: [linguistic_features, tfidf_q_correct],
        LING_AND_READ_AND_R2DE_Q_ALL: [linguistic_features, tfidf_q_all],
        W2V_Q_ONLY_AND_LING_AND_R2DE_Q_ONLY: [w2v_q_only, linguistic_features, tfidf_q_only],
        W2V_Q_ONLY_AND_LING_AND_R2DE_Q_CORRECT: [w2v_q_only, linguistic_features, tfidf_q_correct],
        W2V_Q_ONLY_AND_LING_AND_R2DE_Q_ALL: [w2v_q_only, linguistic_features, tfidf_q_all],
        W2V_Q_ALL_AND_LING_AND_R2DE_Q_ONLY: [w2v_q_all, linguistic_features, tfidf_q_only],
        W2V_Q_ALL_AND_LING_AND_R2DE_Q_CORRECT: [w2v_q_all, linguistic_features, tfidf_q_correct],
        W2V_Q_ALL_AND_LING_AND_R2DE_Q_ALL: [w2v_q_all, linguistic_features, tfidf_q_all],
        W2V_Q_CORRECT_AND_LING_AND_R2DE_Q_ONLY: [w2v_q_correct, linguistic_features, tfidf_q_only],
        W2V_Q_CORRECT_AND_LING_AND_R2DE_Q_CORRECT: [w2v_q_correct, linguistic_features, tfidf_q_correct],
        W2V_Q_CORRECT_AND_LING_AND_R2DE_Q_ALL: [w2v_q_correct, linguistic_features, tfidf_q_all],
    }
    # Returns the feature engineering module
    return FeatureEngineeringModule(dict_feat_eng_components[feature_engineering_config], normalize_method=normalize)


def text2props_get_regression_module_from_config(config, difficulty_range, seed) -> RegressionModule:
    regression_config = config.split('__')[1]
    dict_regression_components = {
        LR: [SklearnRegressionComponent(LinearRegression(), latent_trait_range=difficulty_range)],
        RF: [SklearnRegressionComponent(RandomForestRegressor(random_state=seed), latent_trait_range=difficulty_range)],
    }
    return RegressionModule(dict_regression_components[regression_config])
