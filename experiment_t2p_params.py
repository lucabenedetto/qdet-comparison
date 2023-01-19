from qdet_utils.constants import (
    RACE_PP,
    ARC,
    ARC_BALANCED,
    AM,
    RACE_PP_4K,
    RACE_PP_8K,
    RACE_PP_12K,
)
from qdet_utils.text2props_configs import *

# LIST_DATASET_NAMES = [RACE_PP, ARC, ARC_BALANCED, AM, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K]
# LIST_FEATURE_ENG_CONFIGS = [
#     LING,
#     READ,
#     W2V_Q_ONLY,
#     W2V_Q_ALL,
#     W2V_Q_CORRECT,
#     LING_AND_READ,
#     W2V_Q_ONLY_AND_LING,
#     W2V_Q_ALL_AND_LING,
#     W2V_Q_CORRECT_AND_LING,
#     LING_AND_READ_AND_R2DE_Q_CORRECT,
#     W2V_Q_ONLY_AND_LING_AND_R2DE_Q_CORRECT,
#     LING_AND_READ_AND_R2DE_Q_ALL,
#     W2V_Q_CORRECT_AND_LING_AND_R2DE_Q_ALL,
# ]
# REGRESSION_CONFIG = RF
# RANDOM_SEEDS = [0, 1, 2, 3, 4]

dataset_name = ARC_BALANCED
random_seed = 42
feature_eng_config = READ
regression_config = RF
n_iter = 2
n_jobs = 10
