import numpy as np
import os
import pandas as pd
import pickle

from src.scripts_utils import (
    get_dataframes_text2props,
    evaluate_model,
)
from src.constants import RACE_PP, ARC, ARC_BALANCED, AM, OUTPUT_DIR, DATA_DIR, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K

LIST_DATASET_NAMES = [RACE_PP, AM, ARC, ARC_BALANCED, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K]

for dataset_name in LIST_DATASET_NAMES:

    # dataset-related variables
    df_train, df_test = get_dataframes_text2props(dataset_name)
    discrete_regression = dataset_name in {RACE_PP, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K, ARC, ARC_BALANCED}

    y_true_train = pickle.load(open(os.path.join(DATA_DIR, f'y_true_train_{dataset_name}.p'), 'rb'))
    y_true_dev = pickle.load(open(os.path.join(DATA_DIR, f'y_true_dev_{dataset_name}.p'), 'rb'))
    y_true_test = pickle.load(open(os.path.join(DATA_DIR, f'y_true_test_{dataset_name}.p'), 'rb'))

    print(f'{dataset_name} - Majority')
    output_dir = os.path.join(OUTPUT_DIR, dataset_name)

    if discrete_regression:
        most_popular = pd.DataFrame({'d': y_true_train}).groupby('d').size().reset_index().sort_values(0, ascending=False)['d'].values[0]
        # perform predictions and save them
        y_pred_train = [most_popular] * len(df_train)
        y_pred_test = [most_popular] * len(df_test)

    else:
        mean_diff = np.mean(y_true_train)
        # perform predictions and save them
        y_pred_train = [mean_diff] * len(df_train)
        y_pred_test = [mean_diff] * len(df_test)

    pickle.dump(y_pred_test, open(os.path.join(output_dir, 'predictions_test_majority.p'), 'wb'))
    pickle.dump(y_pred_train, open(os.path.join(output_dir, 'predictions_train_majority.p'), 'wb'))

    evaluate_model(root_string='majority', y_pred_test=y_pred_test, y_pred_train=y_pred_train,
                   y_true_test=y_true_test, y_true_train=y_true_train,
                   output_dir=output_dir, discrete_regression=discrete_regression, compute_correlation=False)
