from src.constants import RACE_PP, ARC, AM


def mapper_race(x):
    if x <= 0.5:
        return 0
    elif x < 1.5:
        return 1
    else:
        return 2


def get_mapper(dataset):
    if dataset == RACE_PP:
        return mapper_race
    else:
        raise NotImplementedError
