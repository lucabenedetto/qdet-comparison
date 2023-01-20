from qdet_utils.experiment import MajorityExperiment
from experiment_majority_params import dataset_name, model_name

if __name__ == "__main__":
    experiment = MajorityExperiment(dataset_name=dataset_name)
    experiment.get_dataset()
    experiment.init_model(None, model_name, None)
    experiment.train(None, None, None, None)
    experiment.predict()
    experiment.evaluate(compute_correlation=False)
