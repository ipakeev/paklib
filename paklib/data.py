from typing import Union

import numpy as np


def to_binary(win: Union[list, np.ndarray], draw: Union[list, np.ndarray] = None) -> np.ndarray:
    win = np.asarray(win, dtype=np.bool)
    y = np.zeros(len(win), dtype=np.float64)
    y[win] = 1.0
    if draw is not None:
        draw = np.asarray(draw, dtype=np.bool)
        y[draw] = 0.5
    return y


def reward(odd: Union[list, np.ndarray], win: Union[list, np.ndarray],
           draw: Union[list, np.ndarray] = None) -> np.ndarray:
    odd = np.asarray(odd, dtype=np.float)
    win = np.asarray(win, dtype=np.bool)
    r = np.full(odd.shape[0], -1.0)
    r[win] = (odd - 1)[win]
    if draw is not None:
        draw = np.asarray(draw, dtype=np.bool)
        r[draw] = 0.0
    return np.around(r, 3)


def sort_on_other_list(list_to_sort: Union[list, np.ndarray], other_list: Union[list, np.ndarray],
                       reverse=False) -> np.ndarray:
    temp = [[i, other_list[i]] for i in range(len(other_list))]
    temp.sort(key=lambda x: x[1], reverse=reverse)
    indices = [i[0] for i in temp]
    return np.array([list_to_sort[i] for i in indices])


def mark_out_sorting(line: Union[list, np.ndarray], max_is_best=False) -> np.ndarray:
    # return rated line, zero rate is best
    worst = -np.inf if max_is_best else np.inf
    line = [i if np.isfinite(i) else worst for i in line]
    temp = list(set(line))
    temp.sort(reverse=max_is_best)  # sort by value
    key = {temp[i]: i for i in range(len(temp))}  # {value: rate}
    rated = np.array([key[value] for value in line])
    return rated
