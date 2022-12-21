import os
import pickle
import random

from src.scripts_utils import get_dataframes_text2props, evaluate_model
from src.constants import RACE_PP, ARC, ARC_GROUPED, AM, OUTPUT_DIR, DATA_DIR, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K

LIST_DATASET_NAMES = [AM]  # [RACE_PP, ARC , RACE_PP_4K, RACE_PP_8K, RACE_PP_12K]

for dataset_name in LIST_DATASET_NAMES:

    # dataset-related variables
    df_train, df_test = get_dataframes_text2props(dataset_name)
    # my_mapper = get_mapper(dataset_name)
    discrete_regression = dataset_name in {RACE_PP, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K, ARC}

    y_true_train = pickle.load(open(os.path.join(DATA_DIR, f'y_true_train_{dataset_name}.p'), 'rb'))
    y_true_dev = pickle.load(open(os.path.join(DATA_DIR, f'y_true_dev_{dataset_name}.p'), 'rb'))
    y_true_test = pickle.load(open(os.path.join(DATA_DIR, f'y_true_test_{dataset_name}.p'), 'rb'))

    for idx in range(5):
        print(f'{dataset_name} - random_{idx}')
        output_dir = os.path.join(OUTPUT_DIR, dataset_name)

        if discrete_regression:
            true_difficulties = list(set(y_true_train))
            # perform predictions and save them
            y_pred_train = [random.choice(true_difficulties) for _ in range(len(df_train))]
            y_pred_test = [random.choice(true_difficulties) for _ in range(len(df_test))]
        else:
            min_diff = min(y_true_train)
            max_diff = max(y_true_train)
            # perform predictions and save them
            y_pred_train = [random.random() * (max_diff - min_diff) + min_diff for _ in range(len(df_train))]
            y_pred_test = [random.random() * (max_diff - min_diff) + min_diff for _ in range(len(df_test))]

        pickle.dump(y_pred_test, open(os.path.join(output_dir, f'predictions_test_random_{idx}.p'), 'wb'))
        pickle.dump(y_pred_train, open(os.path.join(output_dir, f'predictions_train_random_{idx}.p'), 'wb'))

        evaluate_model(root_string=f'random_{idx}', y_pred_test=y_pred_test, y_pred_train=y_pred_train,
                       y_true_test=y_true_test, y_true_train=y_true_train,
                       output_dir=output_dir, discrete_regression=discrete_regression, compute_correlation=True)
