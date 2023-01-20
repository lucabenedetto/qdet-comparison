import os
import pandas as pd

from qdet_utils.constants import (
    RACE_PP, ARC, ARC_BALANCED, AM, OUTPUT_DIR, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K, LIST_TF_ENCODINGS, TF_MODELS, TF_Q_ONLY
)
from qdet_utils.evaluation import evaluate_model
from qdet_utils.difficulty_mapping_methods import get_mapper

LIST_DATASET_NAMES = [RACE_PP, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K, ARC, ARC_BALANCED, AM]

RANDOM_SEEDS = [0, 1, 2, 3, 4]

for dataset_name in LIST_DATASET_NAMES:

    discrete_regression = dataset_name in {RACE_PP, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K, ARC, ARC_BALANCED}
    my_mapper = get_mapper(dataset_name)

    for model in TF_MODELS:
        for encoding in LIST_TF_ENCODINGS:
            if encoding != TF_Q_ONLY and dataset_name == AM:
                continue
            for random_seed in RANDOM_SEEDS:

                filename_train = f'predictions_train_{model}_{encoding}_{random_seed}.csv'
                df_predictions_train = pd.read_csv(os.path.join('data/transformers_predictions', dataset_name, filename_train))
                filename_test = f'predictions_test_{model}_{encoding}_{random_seed}.csv'
                df_predictions_test = pd.read_csv(os.path.join('data/transformers_predictions', dataset_name, filename_test))

                y_true_train = list(df_predictions_train['difficulty'].values)
                y_true_test = list(df_predictions_test['difficulty'].values)

                y_pred_train = list(df_predictions_train[~df_predictions_train['predicted_difficulty'].isna()]['predicted_difficulty'].values)
                y_pred_test = list(df_predictions_test['predicted_difficulty'].values)
                converted_y_pred_train = [my_mapper(x) for x in y_pred_train]
                converted_y_pred_test = [my_mapper(x) for x in y_pred_test]

                output_dir = os.path.join(OUTPUT_DIR, dataset_name, 'seed_' + str(random_seed))
                evaluate_model(model_name=f'{model}_{encoding}',
                               y_pred_test=converted_y_pred_test, y_pred_train=converted_y_pred_train,
                               y_true_test=y_true_test, y_true_train=y_true_train[:len(y_pred_train)],
                               output_dir=output_dir,
                               discrete_regression=discrete_regression,
                               compute_correlation=True)
