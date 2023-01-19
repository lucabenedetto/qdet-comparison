from scipy.stats import randint
from qdet_utils.constants import (
    RACE_PP,
    RACE_PP_4K,
    RACE_PP_8K,
    RACE_PP_12K,
    ARC,
    ARC_BALANCED,
    AM,
)

LIST_DATASET_NAMES = [RACE_PP, ARC, ARC_BALANCED, AM, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K]

dataset_name = ARC_BALANCED
model_name = 'majority'
