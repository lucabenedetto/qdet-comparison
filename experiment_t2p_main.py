from text2props.modules.estimators_from_text import FeatureEngAndRegressionPipeline

from qdet_utils.experiment import Text2propsExperiment
from qdet_utils.text2props_configs import (
    get_config,
    get_regression_module_from_config,
    get_feature_engineering_module_from_config,
    get_dict_params_by_config,
)

from experiment_t2p_params import (
    dataset_name,
    random_seed,
    feature_eng_config,
    regression_config,
    n_iter,
    n_jobs,
)

if __name__ == "__main__":
    experiment = Text2propsExperiment(dataset_name=dataset_name, random_seed=random_seed)
    experiment.get_dataset()

    difficulty_range = experiment.get_difficulty_range(dataset_name)  # TODO this method should probably be somewhere else.
    config = get_config(feature_engineering_config=feature_eng_config, regression_config=regression_config)
    feat_eng_and_regr_pipeline = FeatureEngAndRegressionPipeline(
        get_feature_engineering_module_from_config(config, random_seed),
        get_regression_module_from_config(config, difficulty_range, random_seed)
    )
    experiment.init_model(None, config, feat_eng_and_regr_pipeline)

    dict_params = get_dict_params_by_config(config)
    experiment.train(dict_params, n_iter, n_jobs)
    experiment.predict()
    experiment.evaluate()
