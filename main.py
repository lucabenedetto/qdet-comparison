import os
import pickle

from src.scripts_utils import (
    get_dfs,
    get_true_labels,
    get_mapper,
    get_model_by_config_and_dataset,
    randomized_cv_train,
    get_predictions,
    evaluate_model,
)
from src.constants import RACE_PP, ARC, AM, OUTPUT_DIR
from src.configs import *

dataset = RACE_PP
config = READABILITY_AND_R2DE__LR
print(config)
SEED = 42

output_dir = os.path.join(OUTPUT_DIR, dataset, 'seed_' + str(SEED))

# output_dir = get_output_dir(dataset)
df_train, df_test = get_dfs(dataset)

# get model
model = get_model_by_config_and_dataset(config, dataset)

# train with randomized CV and save model
scores = randomized_cv_train(model, df_train, SEED)
pickle.dump(model, open(os.path.join(output_dir, 'model_' + config + '.p'), 'wb'))

# perform predictions and save them
y_pred_train, y_pred_test = get_predictions(model, df_test, df_train)
pickle.dump(y_pred_test, open(os.path.join(output_dir, 'predictions_test_' + config + '.p'), 'wb'))
pickle.dump(y_pred_train, open(os.path.join(output_dir, 'predictions_train_' + config + '.p'), 'wb'))

my_mapper = get_mapper(dataset)
converted_y_pred_test = [my_mapper(x) for x in y_pred_test]
converted_y_pred_train = [my_mapper(x) for x in y_pred_train]

# TODO this line below might be done once and for all in the data preparation steps
y_true_test, y_true_train = get_true_labels(df_train, df_test, dataset)

evaluate_model(config, converted_y_pred_test, converted_y_pred_train, y_true_test, y_true_train, output_dir)
