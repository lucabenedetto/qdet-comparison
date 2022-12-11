import os
from sklearn.model_selection import RandomizedSearchCV
import pickle
from scipy.stats import randint

from r2de.encoding import get_encoded_texts
from r2de.model import get_model

from src.constants import RACE_PP, ARC, AM, OUTPUT_DIR, DATA_DIR, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K
from src.scripts_utils import get_predictions_r2de, evaluate_model, get_mapper, get_dataframes_r2de

dataset_name = RACE_PP_4K
RANDOM_SEEDS = [0, 1, 2, 3, 4]
ENCODING_IDX = 0
N_ITER = 50
N_JOBS = -1
CV = 5

for dataset_name in [RACE_PP_8K, RACE_PP_12K, RACE_PP]:

    df_train, df_test = get_dataframes_r2de(dataset_name)
    my_mapper = get_mapper(dataset_name)
    discrete_regression = dataset_name in {RACE_PP, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K, ARC}

    y_true_train = pickle.load(open(os.path.join(DATA_DIR, f'y_true_train_{dataset_name}.p'), 'rb'))
    y_true_dev = pickle.load(open(os.path.join(DATA_DIR, f'y_true_dev_{dataset_name}.p'), 'rb'))
    y_true_test = pickle.load(open(os.path.join(DATA_DIR, f'y_true_test_{dataset_name}.p'), 'rb'))

    x_train, x_test = get_encoded_texts(ENCODING_IDX, df_train, df_test)
    param_distribution = {
        'tfidf__max_features': randint(100, 2000),
        'tfidf__max_df': [0.8, 0.85, 0.9, 0.95, 1.0],
        'tfidf__min_df': [1, 0.05, 0.1, 0.15, 0.2],
        'regressor__max_depth': randint(2, 50),
        'regressor__n_estimators': randint(2, 200),
    }

    for random_seed in RANDOM_SEEDS:
        print(f'{dataset_name} - R2DE - encoding {ENCODING_IDX} - seed_{random_seed}')

        output_dir = os.path.join(OUTPUT_DIR, dataset_name, 'seed_' + str(random_seed))

        model = get_model()
        random_search = RandomizedSearchCV(model, param_distribution, n_iter=N_ITER, cv=CV, n_jobs=N_JOBS, random_state=random_seed)

        random_search.fit(x_train, y_true_train)

        best_model = random_search.best_estimator_
        best_scores = random_search.best_score_  # TODO check this (whether it is the same meaning as in T2P)

        # save best model
        pickle.dump(best_model, open(os.path.join(output_dir, f'model_r2de_encoding_{ENCODING_IDX}.p'), 'wb'))

        # perform predictions and save them
        y_pred_train, y_pred_test = get_predictions_r2de(best_model, x_train, x_test)
        pickle.dump(y_pred_test, open(os.path.join(output_dir, f'predictions_test_r2de_encoding_{ENCODING_IDX}.p'), 'wb'))
        pickle.dump(y_pred_train, open(os.path.join(output_dir, f'predictions_train_r2de_encoding_{ENCODING_IDX}.p'), 'wb'))

        converted_y_pred_test = [my_mapper(x) for x in y_pred_test]
        converted_y_pred_train = [my_mapper(x) for x in y_pred_train]

        evaluate_model(
            f'r2de_encoding_{ENCODING_IDX}', converted_y_pred_test, converted_y_pred_train, y_true_test, y_true_train,
            output_dir, discrete_regression=discrete_regression
        )