from qdet_utils.experiment import R2deExperiment
from experiment_r2de_params import (
    param_distribution,
    dataset_name,
    encoding_idx,
    model_name,
    random_seed,
    n_iter,
    n_jobs,
)


def main_r2de():
    experiment = R2deExperiment(dataset_name=dataset_name, random_seed=random_seed)
    experiment.get_dataset()
    experiment.init_model(model_name=model_name, encoding_idx=encoding_idx)
    experiment.train(dict_params=param_distribution, n_iter=n_iter, n_jobs=n_jobs)
    experiment.predict()
    experiment.evaluate()


if __name__ == "__main__":
    main_r2de()
