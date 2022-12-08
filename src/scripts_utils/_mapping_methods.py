from src.constants import RACE_PP, ARC, AM, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K


def mapper_race(x):
    if x <= 0.5:
        return 0
    elif x < 1.5:
        return 1
    else:
        return 2


def get_mapper(dataset):
    if dataset in {RACE_PP, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K}:
        return mapper_race
    else:
        raise NotImplementedError
