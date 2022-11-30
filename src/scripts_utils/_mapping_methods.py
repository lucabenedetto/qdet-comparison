def my_mapper(x):
    if x <= 0.5:
        return 0
    elif x < 1.5:
        return 1
    else:
        return 2


def get_mapper():
    return my_mapper
