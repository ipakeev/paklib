import numpy as np


def to_binary(win, draw=None):
    win = np.asarray(win)
    y = np.zeros(len(win), dtype=np.float64)
    y[win] = 1.0
    if draw is not None:
        draw = np.asarray(draw)
        y[draw] = 0.5
    return y


def reward(odd, win, draw=None):
    odd = np.asarray(odd)
    win = np.asarray(win)
    r = np.full(odd.shape[0], -1.0)
    r[win] = (odd - 1)[win]
    if draw is not None:
        draw = np.asarray(draw)
        r[draw] = 0.0
    return r


def sort_on_other_list(list_to_sort, other_list, reverse=False):
    temp = [[i, other_list[i]] for i in range(len(other_list))]
    temp.sort(key=lambda x: x[1], reverse=reverse)
    indices = [i[0] for i in temp]
    return np.array([list_to_sort[i] for i in indices])


def mark_out_sorting(line, max_is_best=False):
    # return rated line, zero rate is best
    worst = -np.inf if max_is_best else np.inf
    line = [i if np.isfinite(i) else worst for i in line]
    temp = list(set(line))
    temp.sort(reverse=max_is_best)  # sort by value
    key = {temp[i]: i for i in range(len(temp))}  # {value: rate}
    rated = np.array([key[value] for value in line])
    return rated
