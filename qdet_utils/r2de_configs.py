from scipy.stats import randint
from qdet_utils.constants import (
    ARC,
    ARC_BALANCED,
)


def r2de_get_dict_params_by_dataset(dataset_name):
    param_distribution = {'tfidf__max_features': randint(100, 2000),
                          'regressor__max_depth': randint(2, 50),
                          'regressor__n_estimators': randint(2, 200), }
    if dataset_name in {ARC, ARC_BALANCED}:
        param_distribution['tfidf__max_df'] = [0.95, 1.0]
        param_distribution['tfidf__min_df'] = [1, 0.05, 0.1]
    else:
        param_distribution['tfidf__max_df'] = [0.8, 0.85, 0.9, 0.95, 1.0]
        param_distribution['tfidf__min_df'] = [1, 0.05, 0.1, 0.15, 0.2]
    return param_distribution
