import os
import pickle
from src.scripts_utils import (
    get_dataframes,
    get_mapper,
    get_model_by_config_and_dataset,
    randomized_cv_train,
    get_predictions,
    evaluate_model,
)
from src.constants import RACE_PP, ARC, AM, OUTPUT_DIR
from src.configs import *
from src.constants import DATA_DIR

dataset_name = RACE_PP
list_feature_eng_configs = [W2V_Q_ALL_AND_LING]
regression_config = LR
# done (with LR): [READ, W2V_Q_ALL, W2V_Q_CORRECT, W2V_Q_ONLY, LING, LING_AND_READ, W2V_Q_ONLY_AND_LING, W2V_Q_CORRECT_AND_LING]
# done (with RF): []
random_seeds = [4]  # [0, 1, 2, 3, 4]
N_ITER = 20

for feature_eng_config in list_feature_eng_configs:

    # dataset-related variables
    df_train, df_test = get_dataframes(dataset_name)
    my_mapper = get_mapper(dataset_name)
    discrete_regression = dataset_name in {RACE_PP, ARC}

    y_true_train = pickle.load(open(os.path.join(DATA_DIR, f'y_true_train_{dataset_name}.p'), 'rb'))
    y_true_dev = pickle.load(open(os.path.join(DATA_DIR, f'y_true_dev_{dataset_name}.p'), 'rb'))
    y_true_test = pickle.load(open(os.path.join(DATA_DIR, f'y_true_test_{dataset_name}.p'), 'rb'))

    # config-related variables
    config = get_config(feature_engineering_config=feature_eng_config, regression_config=regression_config)
    dict_params = get_dict_params_by_config(config)

    for random_seed in random_seeds:
        print(f' {config} - seed_{random_seed}')

        output_dir = os.path.join(OUTPUT_DIR, dataset_name, 'seed_' + str(random_seed))

        # get model
        model = get_model_by_config_and_dataset(config, dataset_name, random_seed)

        # train with randomized CV and save model
        scores = randomized_cv_train(model, dict_params, df_train, random_seed, n_iter=N_ITER)
        pickle.dump(model, open(os.path.join(output_dir, 'model_' + config + '.p'), 'wb'))

        # perform predictions and save them
        y_pred_train, y_pred_test = get_predictions(model, df_train, df_test)
        pickle.dump(y_pred_test, open(os.path.join(output_dir, 'predictions_test_' + config + '.p'), 'wb'))
        pickle.dump(y_pred_train, open(os.path.join(output_dir, 'predictions_train_' + config + '.p'), 'wb'))

        converted_y_pred_test = [my_mapper(x) for x in y_pred_test]
        converted_y_pred_train = [my_mapper(x) for x in y_pred_train]

        evaluate_model(config, converted_y_pred_test, converted_y_pred_train, y_true_test, y_true_train, output_dir, discrete_regression=discrete_regression)
