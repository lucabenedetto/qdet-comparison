from qdet_utils.experiment import RandomExperiment
from experiment_random_params import dataset_name, model_name


def random_main():
    experiment = RandomExperiment(dataset_name=dataset_name, random_seed=42)
    experiment.get_dataset()
    experiment.init_model(None, model_name, None)
    experiment.train(None, None, None, None)
    experiment.predict()
    experiment.evaluate()


if __name__ == "__main__":
    random_main()
