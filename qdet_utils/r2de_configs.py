from typing import Dict, Any, Optional
from scipy.stats import randint
from qdet_utils.constants import (
    ARC,
    ARC_BALANCED,
)


def r2de_get_dict_params(dataset_name: str, param_distribution: Optional[Dict[str, Any]] = None):
    if param_distribution is not None:
        return param_distribution
    if dataset_name in {ARC, ARC_BALANCED}:
        return {
            'tfidf__max_features': randint(100, 2000),
            'regressor__max_depth': randint(2, 50),
            'regressor__n_estimators': randint(2, 200),
            'tfidf__max_df': [0.95, 1.0],
            'tfidf__min_df': [1, 0.05, 0.1],
        }
    else:
        return {
            'tfidf__max_features': randint(100, 2000),
            'regressor__max_depth': randint(2, 50),
            'regressor__n_estimators': randint(2, 200),
            'tfidf__max_df': [0.8, 0.85, 0.9, 0.95, 1.0],
            'tfidf__min_df': [1, 0.05, 0.1, 0.15, 0.2],
        }
