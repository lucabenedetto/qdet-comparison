from src.constants import RACE_PP, ARC, AM, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K


def mapper_race(x):
    if x <= 0.5:
        return 0
    elif x < 1.5:
        return 1
    else:
        return 2


def mapper_arc_without_balancing(x):
    if x <= 3:
        return 3
    elif x >= 9:
        return 9
    else:
        return round(x)


def get_mapper(dataset: str, balancing: bool = False):
    if dataset in {RACE_PP, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K}:
        return mapper_race
    if dataset == ARC and not balancing:
        return mapper_arc_without_balancing
    else:
        raise NotImplementedError
