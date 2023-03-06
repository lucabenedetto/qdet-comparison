import yaml
from text2props.modules.estimators_from_text import FeatureEngAndRegressionPipeline

from qdet_utils.experiment import (
    MajorityExperiment,
    RandomExperiment,
    R2deExperiment,
    Text2propsExperiment,
)
from qdet_utils.text2props_configs import (
    text2props_get_config,
    text2props_get_regression_module_from_config,
    text2props_get_feature_engineering_module_from_config,
    text2props_get_dict_params_by_config,
)
from qdet_utils.r2de_configs import r2de_get_dict_params_by_dataset


def main(experiment_config):
    model_type = experiment_config['model_type']
    dataset_name = experiment_config['dataset_name']

    if model_type == 'majority':
        model_name = experiment_config['model_name'] if experiment_config['model_name'] != 'None' else 'majority'
        experiment = MajorityExperiment(dataset_name=dataset_name)
        experiment.get_dataset()
        experiment.init_model(None, model_name, None)
        experiment.train(None, None, None, None)
        experiment.predict()
        experiment.evaluate(compute_correlation=False)
        return

    random_seed = experiment_config['random_seed'] if experiment_config['random_seed'] != 'None' else None

    if model_type == 'random':
        model_name = experiment_config['model_name'] if experiment_config['model_name'] != 'None' else 'random'
        experiment = RandomExperiment(dataset_name=dataset_name, random_seed=random_seed)
        experiment.get_dataset()
        experiment.init_model(None, model_name, None)
        experiment.train(None, None, None, None)
        experiment.predict()
        experiment.evaluate()
        return

    n_iter = experiment_config['n_iter']
    n_jobs = experiment_config['n_jobs']
    cv = experiment_config['cv']

    if model_type == 'r2de':
        encoding_idx = experiment_config['encoding_idx']
        model_name = experiment_config['model_name'] if experiment_config['model_name'] != 'None' else f'r2de_encoding{encoding_idx}'
        experiment = R2deExperiment(dataset_name=dataset_name, random_seed=random_seed)
        experiment.get_dataset()
        experiment.init_model(model_name=model_name, encoding_idx=encoding_idx)
        param_distribution = r2de_get_dict_params_by_dataset(dataset_name)
        experiment.train(dict_params=param_distribution, n_iter=n_iter, n_jobs=n_jobs, cv=cv)
        experiment.predict()
        experiment.evaluate()
        return

    if model_type == 'text2props':
        feature_eng_config = experiment_config['feature_eng_config']
        regression_config = experiment_config['regression_config']
        experiment = Text2propsExperiment(dataset_name=dataset_name, random_seed=random_seed)
        experiment.get_dataset()
        difficulty_range = experiment.get_difficulty_range(dataset_name)  # TODO this should probably be somewhere else.
        model_config = text2props_get_config(feature_engineering_config=feature_eng_config, regression_config=regression_config)
        feat_eng_and_regr_pipeline = FeatureEngAndRegressionPipeline(
            text2props_get_feature_engineering_module_from_config(model_config, random_seed),
            text2props_get_regression_module_from_config(model_config, difficulty_range, random_seed)
        )
        model_name = experiment_config['model_name'] if experiment_config['model_name'] != 'None' else model_config
        experiment.init_model(model_name=model_name, feature_eng_and_regression_pipeline=feat_eng_and_regr_pipeline)
        dict_params = text2props_get_dict_params_by_config(model_config)
        experiment.train(dict_params, n_iter, n_jobs, cv=cv)
        experiment.predict()
        experiment.evaluate()
        return

    print("Unknown model_type!")


if __name__ == "__main__":
    config = yaml.safe_load(open('experiment_config.yaml', 'r'))
    main(config)
