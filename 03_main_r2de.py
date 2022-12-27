import os
import pickle
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

from r2de.encoding import get_encoded_texts
from r2de.model import get_model

from src.constants import RACE_PP, ARC, ARC_BALANCED, AM, OUTPUT_DIR, DATA_DIR, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K
from src.scripts_utils import get_predictions_r2de, evaluate_model, get_mapper, get_dataframes_r2de

LIST_DATASET_NAMES = [RACE_PP, ARC, ARC_BALANCED, AM, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K]
RANDOM_SEEDS = [0, 1, 2, 3, 4]
LIST_ENCODING_IDX = [0, 1, 2]
N_ITER = 50
N_JOBS = 8
CV = 5

PARAM_DISTRIBUTION_RACE = {
    'tfidf__max_features': randint(100, 2000),
    'tfidf__max_df': [0.8, 0.85, 0.9, 0.95, 1.0],
    'tfidf__min_df': [1, 0.05, 0.1, 0.15, 0.2],
    'regressor__max_depth': randint(2, 50),
    'regressor__n_estimators': randint(2, 200),
}
PARAM_DISTRIBUTION_ARC = {
    'tfidf__max_features': randint(100, 2000),
    'tfidf__max_df': [0.95, 1.0],
    'tfidf__min_df': [1, 0.05, 0.1],
    'regressor__max_depth': randint(2, 50),
    'regressor__n_estimators': randint(2, 200),
}

PARAM_DISTRIBUTION = PARAM_DISTRIBUTION_ARC

for dataset in LIST_DATASET_NAMES:

    df_train, df_test = get_dataframes_r2de(dataset)
    my_mapper = get_mapper(dataset)
    discrete_regression = dataset in {RACE_PP, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K, ARC, ARC_BALANCED}

    y_true_train = pickle.load(open(os.path.join(DATA_DIR, f'y_true_train_{dataset}.p'), 'rb'))
    y_true_dev = pickle.load(open(os.path.join(DATA_DIR, f'y_true_dev_{dataset}.p'), 'rb'))
    y_true_test = pickle.load(open(os.path.join(DATA_DIR, f'y_true_test_{dataset}.p'), 'rb'))

    for encoding_idx in LIST_ENCODING_IDX:

        x_train, x_test = get_encoded_texts(encoding_idx, df_train, df_test)

        for random_seed in RANDOM_SEEDS:
            print(f'{dataset} - R2DE - encoding {encoding_idx} - seed_{random_seed}')

            output_dir = os.path.join(OUTPUT_DIR, dataset, 'seed_' + str(random_seed))

            model = get_model()
            random_search = RandomizedSearchCV(model, PARAM_DISTRIBUTION, n_iter=N_ITER, cv=CV, n_jobs=N_JOBS, random_state=random_seed)

            random_search.fit(x_train, y_true_train)

            best_model = random_search.best_estimator_
            best_scores = random_search.best_score_

            # save best model
            pickle.dump(best_model, open(os.path.join(output_dir, f'model_r2de_encoding_{encoding_idx}.p'), 'wb'))

            # perform predictions and save them
            y_pred_train, y_pred_test = get_predictions_r2de(best_model, x_train, x_test)
            pickle.dump(y_pred_test, open(os.path.join(output_dir, f'predictions_test_r2de_encoding_{encoding_idx}.p'), 'wb'))
            pickle.dump(y_pred_train, open(os.path.join(output_dir, f'predictions_train_r2de_encoding_{encoding_idx}.p'), 'wb'))

            converted_y_pred_test = [my_mapper(x) for x in y_pred_test]
            converted_y_pred_train = [my_mapper(x) for x in y_pred_train]

            evaluate_model(
                root_string=f'r2de_encoding_{encoding_idx}',
                y_pred_test=converted_y_pred_test, y_pred_train=converted_y_pred_train,
                y_true_test=y_true_test, y_true_train=y_true_train,
                output_dir=output_dir, discrete_regression=discrete_regression
            )
