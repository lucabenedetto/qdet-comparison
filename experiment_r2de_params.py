from scipy.stats import randint
from qdet_utils.constants import (
    RACE_PP,
    RACE_PP_4K,
    RACE_PP_8K,
    RACE_PP_12K,
    ARC,
    ARC_BALANCED,
    AM,
)

LIST_DATASET_NAMES = [RACE_PP, ARC, ARC_BALANCED, AM, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K]
RANDOM_SEEDS = [0, 1, 2, 3, 4]
LIST_ENCODING_IDX = [0, 1, 2]
CV = 5

PARAM_DISTRIBUTION_RACE = {
    'tfidf__max_features': randint(100, 2000),
    'tfidf__max_df': [0.8, 0.85, 0.9, 0.95, 1.0],
    'tfidf__min_df': [1, 0.05, 0.1, 0.15, 0.2],
    'regressor__max_depth': randint(2, 50),
    'regressor__n_estimators': randint(2, 200),
}
PARAM_DISTRIBUTION_ARC = {
    'tfidf__max_features': randint(100, 2000),
    'tfidf__max_df': [0.95, 1.0],
    'tfidf__min_df': [1, 0.05, 0.1],
    'regressor__max_depth': randint(2, 50),
    'regressor__n_estimators': randint(2, 200),
}

param_distribution = PARAM_DISTRIBUTION_ARC
dataset_name = ARC_BALANCED
encoding_idx = 0
model_name = f'r2de_encoding_{encoding_idx}'
random_seed = 42
n_iter = 4
n_jobs = 8
