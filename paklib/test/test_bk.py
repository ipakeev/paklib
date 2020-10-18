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
