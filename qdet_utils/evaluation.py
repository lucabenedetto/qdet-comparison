import os
import pandas as pd
from typing import List
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,  # both MSE and RMSE
    r2_score,
)
from scipy.stats import (
    spearmanr,
    pearsonr,  # this assumes a gaussian distribution for both true and estimated values
)

MAE = 'mean_absolute_error'
RMSE = 'root_mean_squared_error'
R2 = 'r2_score'
SPEARMAN_R = 'spearman_rho'
PEARSON_R = 'pearson_rho'  # only AM
F1 = 'f1_score'  # not ideal for QDET (doesn't consider difficulty ranking, only classification results)
BAL_ACC = 'balanced_accuracy'  # not ideal for QDET (doesn't consider difficulty ranking, only classification results)

METRICS = [MAE, RMSE, R2, SPEARMAN_R, PEARSON_R, F1, BAL_ACC]


def get_metrics(y_true, y_pred, discrete_regression: bool = False, compute_correlation: bool = True):
    metrics = dict()
    metrics[MAE] = mean_absolute_error(y_true, y_pred)
    metrics[RMSE] = mean_squared_error(y_true, y_pred, squared=False)
    metrics[R2] = r2_score(y_true, y_pred)
    if compute_correlation:
        metrics[SPEARMAN_R] = spearmanr(y_true, y_pred).correlation
        metrics[PEARSON_R] = pearsonr(y_true, y_pred).statistic
    else:
        metrics[SPEARMAN_R] = None
        metrics[PEARSON_R] = None
    if discrete_regression:
        metrics[F1] = f1_score(y_true, y_pred, average='weighted')
        metrics[BAL_ACC] = balanced_accuracy_score(y_true, y_pred)
    else:
        metrics[F1] = None
        metrics[BAL_ACC] = None
    return metrics


def evaluate_model(
        root_string: str,
        y_pred_test: List[float],
        y_pred_train: List[float],
        y_true_test: List[float],
        y_true_train: List[float],
        output_dir: str,
        discrete_regression: bool = False,
        compute_correlation: bool = True,
):

    metrics_test = get_metrics(
        y_true_test, y_pred_test, discrete_regression=discrete_regression, compute_correlation=compute_correlation
    )
    metrics_train = get_metrics(
        y_true_train, y_pred_train, discrete_regression=discrete_regression, compute_correlation=compute_correlation
    )

    metrics_dict = dict()
    for metric in metrics_test.keys():
        metrics_dict['test_' + metric] = metrics_test[metric]
        metrics_dict['train_' + metric] = metrics_train[metric]
    out_df = pd.DataFrame([metrics_dict])
    out_df.to_csv(os.path.join(output_dir, 'eval_metrics_' + root_string + '.csv'), index=False)
