from src.constants import RACE_PP, ARC, ARC_BALANCED, AM, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K


def mapper_race(x):
    if x <= 0.5:
        return 0
    elif x < 1.5:
        return 1
    else:
        return 2


def mapper_am(x):
    return x


def mapper_arc(x):
    if x < 3.5:
        return 3
    elif x >= 8.5:
        return 9
    else:
        return round(x)


def get_mapper(dataset: str):
    if dataset in {RACE_PP, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K}:
        return mapper_race
    if dataset in {ARC, ARC_BALANCED}:
        return mapper_arc
    if dataset == AM:
        return mapper_am
    else:
        raise NotImplementedError
