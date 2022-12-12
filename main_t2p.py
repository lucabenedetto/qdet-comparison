import nltk
import os
import pickle

from src.scripts_utils import (
    get_dataframes_text2props,
    get_mapper,
    get_text2props_model_by_config_and_dataset,
    text2props_randomized_cv_train,
    get_predictions_text2props,
    evaluate_model,
)
from src.constants import RACE_PP, ARC, AM, OUTPUT_DIR, DATA_DIR, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K
from src.configs import *

nltk.download('averaged_perceptron_tagger')

LIST_DATASET_NAMES = [RACE_PP]
LIST_FEATURE_ENG_CONFIGS = [W2V_Q_ALL, W2V_Q_CORRECT, W2V_Q_ONLY]
REGRESSION_CONFIG = LR
RANDOM_SEEDS = [0, 1, 2, 3, 4]
N_ITER = 20
N_JOBS = 10

for dataset_name in LIST_DATASET_NAMES:

    # dataset-related variables
    df_train, df_test = get_dataframes_text2props(dataset_name)
    my_mapper = get_mapper(dataset_name)
    discrete_regression = dataset_name in {RACE_PP, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K, ARC}

    y_true_train = pickle.load(open(os.path.join(DATA_DIR, f'y_true_train_{dataset_name}.p'), 'rb'))
    y_true_dev = pickle.load(open(os.path.join(DATA_DIR, f'y_true_dev_{dataset_name}.p'), 'rb'))
    y_true_test = pickle.load(open(os.path.join(DATA_DIR, f'y_true_test_{dataset_name}.p'), 'rb'))

    for feature_eng_config in LIST_FEATURE_ENG_CONFIGS:

        # config-related variables
        config = get_config(feature_engineering_config=feature_eng_config, regression_config=REGRESSION_CONFIG)
        dict_params = get_dict_params_by_config(config)

        for random_seed in RANDOM_SEEDS:
            print(f' {config} - seed_{random_seed}')

            output_dir = os.path.join(OUTPUT_DIR, dataset_name, 'seed_' + str(random_seed))

            # get model
            model = get_text2props_model_by_config_and_dataset(config, dataset_name, random_seed)

            # train with randomized CV and save model
            scores = text2props_randomized_cv_train(model, dict_params, df_train, random_seed, n_iter=N_ITER, n_jobs=N_JOBS)
            pickle.dump(model, open(os.path.join(output_dir, 'model_' + config + '.p'), 'wb'))

            # perform predictions and save them
            y_pred_train, y_pred_test = get_predictions_text2props(model, df_train, df_test)
            pickle.dump(y_pred_test, open(os.path.join(output_dir, 'predictions_test_' + config + '.p'), 'wb'))
            pickle.dump(y_pred_train, open(os.path.join(output_dir, 'predictions_train_' + config + '.p'), 'wb'))

            converted_y_pred_test = [my_mapper(x) for x in y_pred_test]
            converted_y_pred_train = [my_mapper(x) for x in y_pred_train]

            evaluate_model(config, converted_y_pred_test, converted_y_pred_train, y_true_test, y_true_train,
                           output_dir, discrete_regression=discrete_regression)
