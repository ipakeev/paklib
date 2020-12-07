from paklib import bk


def test_get_payout():
    assert bk.get_payout([1.9, 1.9]) == 0.95


def test_get_proba():
    assert list(bk.get_proba([1.9, 1.9])) == [0.5, 0.5]


def test_get_odd():
    assert list(bk.get_odd([0.5, 0.5], payout=0.95)) == [1.9, 1.9]


def test_second_odd():
    assert bk.second_odd(1.9, payout=0.95) == 1.9


def test_probability():
    p = bk.probability(1052, 1235)
    assert 0.0 < p < 0.5


def test_describe():
    import numpy as np
    win = np.array([True, True, False, False, True, False])
    draw = np.array([False, False, False, False, False, True])
    odds = np.array([2.0, 2.5, 2.2, 2.3, 2.0, 2.0])
    assert bk.describe(win, draw=draw).__repr__() == '3/1/2 (5), p=0.600'
    assert bk.describe(win, draw=draw, odds=odds).__repr__() == '3/1/2 (5), b=1.5, p=0.600, roi=0.300'


def test_bank():
    import datetime
    import pytest
    b = bk.Bank()
    b.stake(datetime.datetime(2020, 10, 1, 17, 00), 'id0', 'name0', 1.0, 2.0, True, False)
    b.stake(datetime.datetime(2020, 10, 1, 15, 00), 'id1', 'name1', 1.0, 2.5, True, False)
    b.stake(datetime.datetime(2020, 10, 1, 20, 00), 'id2', 'name2', 1.0, 2.2, False, False)
    b.stake(datetime.datetime(2020, 10, 3, 15, 00), 'id3', 'name3', 1.0, 2.3, False, False)
    b.stake(datetime.datetime(2020, 10, 5, 22, 00), 'id4', 'name4', 1.0, 2.0, True, False)
    b.stake(datetime.datetime(2020, 10, 7, 7, 00), 'id5', 'name5', 1.0, 2.0, False, True)
    assert b.describe().__repr__() == '3/1/2 (5), b=1.5, p=0.600, roi=0.300'
    with pytest.raises(AssertionError):
        b.stake(datetime.datetime(2020, 10, 7, 7, 00), 'id5', 'name5', 1.0, 2.0, True, True)
    b.plot()
