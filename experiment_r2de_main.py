from qdet_utils.experiment import R2deExperiment
import yaml
from scipy.stats import randint


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


def main_r2de():
    config = yaml.safe_load(open('experiment_r2de_config.yaml', 'r'))
    dataset_name = config['dataset_name']
    encoding_idx = config['encoding_idx']
    model_name = config['model_name']
    random_seed = config['random_seed']
    n_iter = config['n_iter']
    n_jobs = config['n_jobs']

    experiment = R2deExperiment(dataset_name=dataset_name, random_seed=random_seed)
    experiment.get_dataset()
    experiment.init_model(model_name=model_name, encoding_idx=encoding_idx)
    experiment.train(dict_params=PARAM_DISTRIBUTION, n_iter=n_iter, n_jobs=n_jobs)
    experiment.predict()
    experiment.evaluate()


if __name__ == "__main__":
    main_r2de()
