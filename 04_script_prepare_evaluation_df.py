import numpy as np
import os
import pandas as pd

from src.constants import RACE_PP, ARC, ARC_GROUPED, AM, OUTPUT_DIR, DATA_DIR, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K
from src.configs import *
from src.scripts_utils import METRICS


LIST_DATASET_NAMES = [ARC]
LIST_FEATURE_ENG_CONFIGS = [
    LING,
    READ,
    W2V_Q_ONLY,
    # W2V_Q_ALL,
    # W2V_Q_CORRECT,
    LING_AND_READ,
    W2V_Q_ONLY_AND_LING,
    # W2V_Q_ALL_AND_LING,
    # W2V_Q_CORRECT_AND_LING,
]
REGRESSION_CONFIG = RF
RANDOM_SEEDS = [0, 1, 2, 3, 4]


def main():
    # work separately on each dataset
    for dataset in LIST_DATASET_NAMES:

        output_df = pd.DataFrame()

        # Models implemented with text2props
        # todo add the other models
        for feature_eng_config in LIST_FEATURE_ENG_CONFIGS:
            config = get_config(feature_engineering_config=feature_eng_config, regression_config=REGRESSION_CONFIG)
            new_row_dict = get_dict_results_for_model(dataset, config)
            output_df = pd.concat([output_df, pd.DataFrame([new_row_dict])], ignore_index=True)

        # models implemented with R2DE
        # todo

        # random
        new_row_dict = get_dict_results_for_model(dataset, 'random')
        output_df = pd.concat([output_df, pd.DataFrame([new_row_dict])], ignore_index=True)

        # majority
        new_row_dict = {col: None for col in output_df.columns}
        local_df = pd.read_csv(os.path.join(OUTPUT_DIR, dataset, f'eval_metrics_majority.csv'))
        for metric in METRICS:
            new_row_dict[f'train_{metric}_mean'] = local_df[f'train_{metric}'].values[0]
            new_row_dict[f'train_{metric}_median'] = local_df[f'train_{metric}'].values[0]
            new_row_dict[f'train_{metric}_std'] = 0
            new_row_dict[f'test_{metric}_mean'] = local_df[f'test_{metric}'].values[0]
            new_row_dict[f'test_{metric}_median'] = local_df[f'test_{metric}'].values[0]
            new_row_dict[f'test_{metric}_std'] = 0
        new_row_dict['model'] = 'majority'
        output_df = pd.concat([output_df, pd.DataFrame([new_row_dict])], ignore_index=True)

        output_df.to_csv(os.path.join(OUTPUT_DIR, dataset, 'general_evaluation.csv'), index=False)


def get_dict_results_for_model(dataset, config):
    new_row_dict = dict()
    for random_seed in RANDOM_SEEDS:
        local_df = pd.read_csv(os.path.join(OUTPUT_DIR, dataset, 'seed_' + str(random_seed), f'eval_metrics_{config}.csv'))
        for metric in METRICS:
            new_row_dict[f'train_seed_{random_seed}_{metric}'] = local_df[f'train_{metric}'].values[0]
            new_row_dict[f'test_seed_{random_seed}_{metric}'] = local_df[f'test_{metric}'].values[0]
    for metric in METRICS:
        list_train_results = [new_row_dict[f'train_seed_{seed}_{metric}'] for seed in RANDOM_SEEDS]
        new_row_dict[f'train_{metric}_mean'] = np.mean(list_train_results)
        new_row_dict[f'train_{metric}_median'] = np.median(list_train_results)
        new_row_dict[f'train_{metric}_std'] = np.std(list_train_results)
        list_test_results = [new_row_dict[f'train_seed_{seed}_{metric}'] for seed in RANDOM_SEEDS]
        new_row_dict[f'test_{metric}_mean'] = np.mean(list_test_results)
        new_row_dict[f'test_{metric}_median'] = np.median(list_test_results)
        new_row_dict[f'test_{metric}_std'] = np.std(list_test_results)
    new_row_dict['model'] = config
    return new_row_dict


main()
