import pytest

from paklib import bk


def test_get_payout():
    assert bk.get_payout([1.9, 1.9]) == 0.95


def test_get_proba():
    assert list(bk.get_proba([1.9, 1.9])) == [0.5, 0.5]
    p = bk.get_proba([1.6, 2.2])
    assert p[0] > p[1]


def test_get_odd():
    assert list(bk.get_odd([0.5, 0.5], payout=0.95)) == [1.9, 1.9]
    odds = bk.get_odd([0.6, 0.4], payout=0.95)
    assert odds[0] < odds[1]


def test_second_odd():
    assert bk.second_odd(1.9, payout=0.95) == 1.9
    assert 1.7 < bk.second_odd(1.7, payout=0.95)


def test_probability():
    p = bk.probability(1052, 1235)
    assert 0.0 < p < 0.5


def test_DNB():
    assert bk.DNB([1.8, 2.0]) == [1.8, 2.0]
    assert bk.DNB([1.9, 1.9], proba=True) == [0.5, 0.5]
    odds = [1.9, 3.5, 4.5]
    assert bk.DNB(odds) == [1.28, 3.03]
    assert bk.DNB(odds, proba=True) == [0.703, 0.297]


def test_DC():
    assert bk.DC([1.8, 2.0]) == [1.8, 2.0, 1.0]
    assert bk.DC([1.9, 1.9], proba=True) == [0.5, 0.5, 1.0]
    odds = [1.9, 3.5, 4.5]
    assert bk.DC(odds) == [1.146, 1.833, 1.243]
    assert bk.DC(odds, proba=True) == [0.785, 0.491, 0.724]


def test_DNB_DC():
    assert bk.DNB_DC([1.8, 2.0]) == [1.8, 2.0, 1.8, 2.0, 1.0]
    assert bk.DNB_DC([1.9, 1.9], proba=True) == [0.5, 0.5, 0.5, 0.5, 1.0]
    odds = [1.9, 3.5, 4.5]
    assert bk.DNB_DC(odds) == [1.28, 3.03, 1.146, 1.833, 1.243]
    assert bk.DNB_DC(odds, proba=True) == [0.703, 0.297, 0.785, 0.491, 0.724]


def test_kelly():
    assert bk.kelly(0.5, 1.8) < 0.0
    assert bk.kelly(0.7, 1.9) > 0.0


def test_stake_size():
    assert bk.stake_size(2.0, to_win=1.0) == 1.0
    assert bk.stake_size(1.5, to_win=1.0) == 2.0


def test_describe():
    import numpy as np
    win = np.array([True, True, False, False, True, False])
    draw = np.array([False, False, False, False, False, True])
    odds = np.array([2.0, 2.5, 2.2, 2.3, 2.0, 2.0])
    assert bk.describe(win, draw=draw).__repr__() == '3/1/2 (5), b=0.0, p=0.600, roi=0.000, score=0.000'

    desc = bk.describe(win, draw=draw, odds=odds, target='bank')
    assert desc.__repr__() == '3/1/2 (5), b=1.5, p=0.600, roi=0.300, score=0.897'

    assert desc > bk.describe(np.array([True]), odds=np.array([2.0]))
    assert not desc > bk.describe(np.array([True]), odds=np.array([5.0]))
    assert desc < bk.describe(np.array([True]), odds=np.array([5.0]))
    assert not desc < bk.describe(np.array([True]), odds=np.array([2.0]))

    assert desc > 1.0
    assert desc >= 1.5
    assert desc <= 1.5
    assert desc < 2.0

    with pytest.raises(AssertionError):
        _ = desc > bk.describe(np.array([True]), odds=np.array([2.0]), target='roi')


def test_bank():
    import datetime
    import pytest
    b = bk.Bank()
    b.stake(datetime.datetime(2020, 10, 1, 17, 00), 1, 'name0', 2.0, True, False)
    b.stake(datetime.datetime(2020, 10, 1, 15, 00), 1, 'name1', 2.5, True, False)
    b.stake(datetime.datetime(2020, 10, 1, 20, 00), 1, 'name2', 2.2, False, False)
    b.stake(datetime.datetime(2020, 10, 3, 15, 00), 1, 'name3', 2.3, False, False)
    b.stake(datetime.datetime(2020, 10, 5, 22, 00), 2, 'name4', 2.0, True, False)
    b.stake(datetime.datetime(2020, 10, 7, 7, 00), 2, 'name5', 2.0, False, True)
    assert b.describe().__repr__() == '3/1/2 (5), b=1.5, p=0.600, roi=0.300, score=0.897'
    assert b.describe(league_id=1).__repr__() == '2/0/2 (4), b=0.5, p=0.500, roi=0.125, score=0.218'
    with pytest.raises(AssertionError):
        b.stake(datetime.datetime(2020, 10, 7, 7, 00), 3, 'name5', 2.0, True, True)
    b.plot()
    # b.plot(save_name='img/1.jpg')
