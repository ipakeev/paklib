import numpy as np
from paklib import data


def test_to_binary():
    src = data.to_binary([True, False, False, True])
    target = np.array([1.0, 0.0, 0.0, 1.0])
    assert np.all(src == target)

    src = data.to_binary([True, False, False, True], draw=[False, True, False, False])
    target = np.array([1.0, 0.5, 0.0, 1.0])
    assert np.all(src == target)


def test_sort_on_other_list():
    assert np.all(data.sort_on_other_list([1, 2, 0, 1, 3], [1, 3, 5, 2, 4]) == np.array([1, 1, 2, 3, 0]))
    assert np.all(data.sort_on_other_list([1, 2, 0, 1, 3], [1, 3, 5, 2, 4], reverse=True) == np.array([0, 3, 2, 1, 1]))


def test_mark_out_sorting():
    assert np.all(data.mark_out_sorting([1, 2, 0, 1, 3], max_is_best=True) == np.array([2, 1, 3, 2, 0]))
    assert np.all(data.mark_out_sorting([1, 2, 0, 1, 3], max_is_best=False) == np.array([1, 2, 0, 1, 3]))
    source = [1, 2, np.nan, np.inf, 0, 1, 3]
    target = np.array([2, 1, 4, 4, 3, 2, 0])
    assert np.all(data.mark_out_sorting(source, max_is_best=True) == target)
    source = [1, 2, np.inf, np.nan, 0, 1, 3]
    target = np.array([1, 2, 4, 4, 0, 1, 3])
    assert np.all(data.mark_out_sorting(source, max_is_best=False) == target)
