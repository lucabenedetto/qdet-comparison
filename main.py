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
feature_eng_config = READ
regression_config = LR
SEED = 42

config = get_config(feature_engineering_config=feature_eng_config, regression_config=regression_config)
print(config)

output_dir = os.path.join(OUTPUT_DIR, dataset_name, 'seed_' + str(SEED))

# output_dir = get_output_dir(dataset)
df_train, df_test = get_dataframes(dataset_name)

# get model
model = get_model_by_config_and_dataset(config, dataset_name, SEED)

# train with randomized CV and save model
dict_params = get_dict_params_by_config(config)
scores = randomized_cv_train(model, dict_params, df_train, SEED)
pickle.dump(model, open(os.path.join(output_dir, 'model_' + config + '.p'), 'wb'))

# perform predictions and save them
y_pred_train, y_pred_test = get_predictions(model, df_train, df_test)
pickle.dump(y_pred_test, open(os.path.join(output_dir, 'predictions_test_' + config + '.p'), 'wb'))
pickle.dump(y_pred_train, open(os.path.join(output_dir, 'predictions_train_' + config + '.p'), 'wb'))

my_mapper = get_mapper(dataset_name)
converted_y_pred_test = [my_mapper(x) for x in y_pred_test]
converted_y_pred_train = [my_mapper(x) for x in y_pred_train]

y_true_train = pickle.load(open(os.path.join(DATA_DIR, f'y_true_train_{dataset_name}.p'), 'rb'))
y_true_dev = pickle.load(open(os.path.join(DATA_DIR, f'y_true_dev_{dataset_name}.p'), 'rb'))
y_true_test = pickle.load(open(os.path.join(DATA_DIR, f'y_true_test_{dataset_name}.p'), 'rb'))

evaluate_model(config, converted_y_pred_test, converted_y_pred_train, y_true_test, y_true_train, output_dir)
