from src.constants import RACE_PP, ARC, AM, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K


def mapper_race(x):
    if x <= 0.5:
        return 0
    elif x < 1.5:
        return 1
    else:
        return 2


def mapper_am(x):
    return x


def mapper_arc_without_grouping(x):
    if x < 3.5:
        return 3
    elif x >= 8.5:
        return 9
    else:
        return round(x)


def mapper_arc_with_grouping(x):
    if x < 4.5:
        return 3
    elif x < 5.5:
        return 5
    elif x < 7.5:
        return 7
    elif x < 8.5:
        return 8
    else:
        return 9
# 3 & 4 - 115+379=494 | 5 - 690 | 6 & 7 - 99+406 = 505 | 8 - 1375 | 9 - 294


def get_mapper(dataset: str, grouping: bool = False):
    if dataset in {RACE_PP, RACE_PP_4K, RACE_PP_8K, RACE_PP_12K}:
        return mapper_race
    if dataset == ARC and not grouping:
        return mapper_arc_without_grouping
    if dataset == ARC and grouping:
        return mapper_arc_with_grouping
    if dataset == AM:
        return mapper_am
    else:
        raise NotImplementedError
