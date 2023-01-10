import numpy as np
import os
import pandas as pd

from src.constants import RACE_PP, ARC, ARC_BALANCED, AM, OUTPUT_DIR, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K
from src.constants import LIST_TF_ENCODINGS, TF_MODELS, TF_Q_ONLY
from src.configs import *
from src.scripts_utils import METRICS


LIST_DATASET_NAMES = [RACE_PP]  # [RACE_PP, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K, ARC, ARC_BALANCED, AM]
LIST_FEATURE_ENG_CONFIGS = [
    # LING,
    # READ,
    # W2V_Q_ONLY,
    # W2V_Q_ALL,
    # W2V_Q_CORRECT,
    # LING_AND_READ,
    # W2V_Q_ONLY_AND_LING,
    # W2V_Q_ALL_AND_LING,
    # W2V_Q_CORRECT_AND_LING,
    # W2V_Q_ONLY_AND_LING_AND_R2DE_Q_ONLY,
    # LING_AND_READ_AND_R2DE_Q_CORRECT,
    W2V_Q_ALL_AND_LING_AND_R2DE_Q_CORRECT,
]
REGRESSION_CONFIG = RF
RANDOM_SEEDS = [0, 1]


def main():
    # work separately on each dataset
    for dataset in LIST_DATASET_NAMES:

        output_df_train = pd.DataFrame()
        output_df_test = pd.DataFrame()

        # Models implemented with text2props
        for feature_eng_config in LIST_FEATURE_ENG_CONFIGS:
            if feature_eng_config in {W2V_Q_ALL, W2V_Q_CORRECT, W2V_Q_ALL_AND_LING, W2V_Q_CORRECT_AND_LING} and dataset == AM:
                continue
            config = get_config(feature_engineering_config=feature_eng_config, regression_config=REGRESSION_CONFIG)
            new_row_dict_train, new_row_dict_test = get_dict_results_for_model(dataset, config)
            output_df_train = pd.concat([output_df_train, pd.DataFrame([new_row_dict_train])], ignore_index=True)
            output_df_test = pd.concat([output_df_test, pd.DataFrame([new_row_dict_test])], ignore_index=True)

        # models implemented with R2DE
        # for encoding in [0, 1, 2]:
        #     if encoding != 0 and dataset == AM:
        #         continue
        #     new_row_dict_train, new_row_dict_test = get_dict_results_for_model(dataset, f'r2de_encoding_{encoding}')
        #     output_df_train = pd.concat([output_df_train, pd.DataFrame([new_row_dict_train])], ignore_index=True)
        #     output_df_test = pd.concat([output_df_test, pd.DataFrame([new_row_dict_test])], ignore_index=True)

        # random
        # new_row_dict_train, new_row_dict_test = get_dict_results_for_model(dataset, 'random')
        # output_df_train = pd.concat([output_df_train, pd.DataFrame([new_row_dict_train])], ignore_index=True)
        # output_df_test = pd.concat([output_df_test, pd.DataFrame([new_row_dict_test])], ignore_index=True)

        # majority
        # new_row_dict_train = {col: None for col in output_df_train.columns}
        # new_row_dict_test = {col: None for col in output_df_test.columns}
        # local_df = pd.read_csv(os.path.join(OUTPUT_DIR, dataset, f'eval_metrics_majority.csv'))
        # for metric in METRICS:
        #     new_row_dict_train[f'train_{metric}_mean'] = local_df[f'train_{metric}'].values[0]
        #     new_row_dict_train[f'train_{metric}_std'] = np.nan
        #     new_row_dict_test[f'test_{metric}_mean'] = local_df[f'test_{metric}'].values[0]
        #     new_row_dict_test[f'test_{metric}_std'] = np.nan
        # new_row_dict_train['model'] = 'majority'
        # new_row_dict_test['model'] = 'majority'
        # output_df_train = pd.concat([output_df_train, pd.DataFrame([new_row_dict_train])], ignore_index=True)
        # output_df_test = pd.concat([output_df_test, pd.DataFrame([new_row_dict_test])], ignore_index=True)

        # transformers
        # for model in TF_MODELS:
        #     for encoding in LIST_TF_ENCODINGS:
        #         if encoding != TF_Q_ONLY and dataset == AM:
        #             continue
        #         new_row_dict_train, new_row_dict_test = get_dict_results_for_model(dataset, f'{model}_{encoding}')
        #         output_df_train = pd.concat([output_df_train, pd.DataFrame([new_row_dict_train])], ignore_index=True)
        #         output_df_test = pd.concat([output_df_test, pd.DataFrame([new_row_dict_test])], ignore_index=True)

        output_df_train.to_csv(os.path.join(OUTPUT_DIR, dataset, 'general_evaluation_train.csv'), index=False)
        output_df_test.to_csv(os.path.join(OUTPUT_DIR, dataset, 'general_evaluation_test.csv'), index=False)


def get_dict_results_for_model(dataset, config):
    new_row_dict_train = dict()
    new_row_dict_test = dict()
    for random_seed in RANDOM_SEEDS:
        local_df = pd.read_csv(os.path.join(OUTPUT_DIR, dataset, 'seed_' + str(random_seed), f'eval_metrics_{config}.csv'))
        for metric in METRICS:
            new_row_dict_train[f'train_seed_{random_seed}_{metric}'] = local_df[f'train_{metric}'].values[0]
            new_row_dict_test[f'test_seed_{random_seed}_{metric}'] = local_df[f'test_{metric}'].values[0]
    for metric in METRICS:
        list_train_results = [new_row_dict_train[f'train_seed_{seed}_{metric}'] for seed in RANDOM_SEEDS]
        new_row_dict_train[f'train_{metric}_mean'] = np.mean(list_train_results)
        new_row_dict_train[f'train_{metric}_std'] = np.std(list_train_results)

        list_test_results = [new_row_dict_test[f'test_seed_{seed}_{metric}'] for seed in RANDOM_SEEDS]
        new_row_dict_test[f'test_{metric}_mean'] = np.mean(list_test_results)
        new_row_dict_test[f'test_{metric}_std'] = np.std(list_test_results)
    new_row_dict_train['model'] = config
    new_row_dict_test['model'] = config
    return new_row_dict_train, new_row_dict_test


main()
