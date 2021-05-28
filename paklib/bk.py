import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd


def get_payout(odd):
    odd = np.asarray(odd)
    return np.around(1 / np.sum(1 / odd), 3)


def get_proba(odd):
    odd = np.asarray(odd)
    payout = get_payout(odd)
    return np.around(payout / odd, 3)


def get_odd(proba, payout=0.9):
    proba = np.maximum(0.01, np.minimum(proba, 0.99))
    odd = payout / proba
    odd = np.maximum(1.01, np.minimum(odd, 10.0))
    return np.around(odd, 3)


def second_odd(odd, payout=0.95):
    proba = 1.0 - payout / odd
    return np.around(payout / proba, 3)


def e_score_mean(scores, er=-0.0065):
    x, y = 0.0, 0.0
    for n, i in enumerate(scores[::-1]):
        e = np.e ** (er * n)
        x += e
        y += i * e
    return y / x


def probability(r1, r2):
    return 1 / (1 + 10 ** ((r2 - r1) / 400))


def exp_probability(r1, r2, alpha=0.0, beta=1.0, k=1.0, b=0.0):
    p = np.exp(alpha + beta * (r1 - r2)) / (1 + np.exp(alpha + beta * (r1 - r2)))
    p = k * p + b
    return p


def elo(r1, r2, real_points, g=None):
    return 20.0 * (real_points - probability(r1, r2))


def elo_goal(r1, r2, real_points, g):
    g = abs(g)
    dg = 1.0 if g <= 1 else 1.5 if g == 2 else ((11.0 + g) / 8)
    return 20.0 * dg * (real_points - probability(r1, r2))


def elo_value(r1, r2, real_points, g):
    dg = abs(g) * 10
    return 20.0 * dg * (real_points - probability(r1, r2))


def DNB(odd, proba=False):
    p = get_proba(odd)
    if len(odd) == 2:
        return list(p) if proba else list(odd)
    al = p[0] + p[-1]
    p = np.around([p[0] / al, p[2] / al], 3)
    if proba:
        return list(p)
    return list(get_odd(p))


def DC(odd, proba=False):
    p = get_proba(odd)
    if len(odd) == 2:
        if proba:
            return list(p) + [1.0]
        else:
            return list(odd) + [1.0]
    p = np.around([p[0] + p[1], p[1] + p[2], p[0] + p[2]], 3)
    if proba:
        return list(p)
    return list(get_odd(p))


def DNB_DC(odd, proba=False):
    p = get_proba(odd)
    if len(odd) == 2:
        if proba:
            return list(p) + list(p) + [1.0]
        else:
            return list(odd) + list(odd) + [1.0]
    al = p[0] + p[-1]
    p = np.around([p[0] / al, p[2] / al, p[0] + p[1], p[1] + p[2], p[0] + p[2]], 3)
    if proba:
        return list(p)
    return list(get_odd(p))


def kelly(pred, odd, mask=None):
    if mask is not None:
        pred = pred[mask]
        odd = odd[mask]
    return (pred * odd - 1) / (odd - 1)


def reward(odd, y):
    r = odd - 1.0
    r[y == 0.0] = -1.0
    r[y == 0.25] = -0.5
    r[y == 0.5] = 0.0
    r[y == 0.75] *= 0.5
    return r


def stake_size(odd, to_win=1.0):
    return to_win / (odd - 1)


def stake_system(tip_min_win, tip_all, win, draw=0, stake=1, odd=1.85, printer=True):
    import itertools

    tips = list(range(tip_all))
    pay = [odd] * win + [1.0] * draw + [0.0] * (tip_all - win - draw)
    combo = list(itertools.combinations(tips, tip_min_win))
    variance = len(combo)
    price = dict([(t, p) for t, p in zip(tips, pay)])
    stakePrice = stake / variance
    plus = -stake
    for c in combo:
        b = 1.0
        for i in c:
            b *= price[i]
        plus += b * stakePrice
    if printer:
        print('({}) / plus: {:+.2f} ({:+.1f}%)'.format(variance, plus, 100 * plus / stake))
    else:
        return plus


def stake_system_optimal(tip_all, mn=2, mx=6, draw=0, stake=1, odd=1.85):
    import matplotlib.pylab as plt

    mx = mx if mx <= tip_all else tip_all
    wins = np.arange(1, tip_all + 1)
    for tipWin in range(mn, mx + 1):
        plus = [stake_system(tipWin, tip_all, i - draw, draw=draw, stake=stake, odd=odd, printer=False)
                for i in wins]
        plt.plot(wins, plus, label='{}/{}'.format(tipWin, tip_all))
    plt.xticks(wins, ['{:.0f}\n{:.0f}%'.format(i, 100 * i / tip_all) for i in wins])
    plt.legend()
    plt.show()


@dataclass
class StakesDescription:

    def __init__(self, win: int, draw: int, lose: int, bank: float = None,
                 cf: float = None, target: str = 'bank'):
        self.win = int(win)
        self.draw = int(draw)
        self.lose = int(lose)
        self.n_stakes = self.win + self.lose

        if self.n_stakes:
            self.proba = self.win / self.n_stakes
        else:
            self.proba = 0.0

        if (bank is not None) and self.n_stakes:
            self.bank = bank if cf is None else bank * cf
            self.roi = bank / self.n_stakes
            self.score = self.bank * (min(0.7, self.proba) ** 0.3) * (min(0.7, self.roi) ** 0.3) if bank > 0 else bank
        else:
            self.bank = 0.0
            self.roi = 0.0
            self.score = 0.0

        self.target = target

    def _check_other(self, other):
        assert self.target == other.target

    def __add__(self, other):
        return StakesDescription(self.win + other.win,
                                 self.draw + other.draw,
                                 self.lose + other.lose,
                                 bank=self.bank + other.bank)

    def __gt__(self, other):
        if isinstance(other, StakesDescription):
            self._check_other(other)
            return self.get(self.target) > other.get(self.target)
        return self.get(self.target) > other

    def __ge__(self, other):
        if isinstance(other, StakesDescription):
            self._check_other(other)
            return self.get(self.target) >= other.get(self.target)
        return self.get(self.target) >= other

    def __lt__(self, other):
        if isinstance(other, StakesDescription):
            self._check_other(other)
            return self.get(self.target) < other.get(self.target)
        return self.get(self.target) < other

    def __le__(self, other):
        if isinstance(other, StakesDescription):
            self._check_other(other)
            return self.get(self.target) <= other.get(self.target)
        return self.get(self.target) <= other

    def __repr__(self):
        return f'{self.win}/{self.draw}/{self.lose} ({self.n_stakes}), ' \
               f'b={self.bank:.1f}, p={self.proba:.3f}, roi={self.roi:.3f}, score={self.score:.3f}'

    def get(self, name: str = None):
        return self.__getattribute__(self.target if name is None else name)


def describe(win, draw=None, odds=None, mask=None, target: str = 'bank') -> StakesDescription:
    assert win.dtype == bool
    draw = draw if draw is not None else np.full(win.shape, False)
    if odds is not None:
        if mask is None:
            mask = ~np.isnan(odds)
        else:
            mask &= ~np.isnan(odds)
    if mask is not None:
        win = win[mask]
        draw = draw[mask]
        if odds is not None:
            odds = odds[mask]

    W = win.sum()
    D = draw.sum()
    L = win.shape[0] - W - D

    if odds is None:
        return StakesDescription(W, D, L, target=target)
    else:
        B = (odds - 1.0)[win].sum() - L
        return StakesDescription(W, D, L, bank=B, target=target)


class Bank(object):

    def __init__(self):
        self.df = pd.DataFrame(columns=['date_time', 'league_id', 'name',
                                        'pred', 'odds', 'win', 'draw', 'plus', 'value'])

    def stake(self, date_time: datetime.datetime, league_id: int, name: str,
              pred: float, odds: float, win: bool, draw: bool = False, value: float = 1.0):
        assert not (win and draw)
        if win:
            plus = value * (odds - 1.0)
        elif draw:
            plus = 0.0
        else:
            plus = -value
        self.df.loc[len(self.df)] = [date_time, league_id, name, pred, odds, win, draw, plus, value]

    def describe(self, mask=None) -> StakesDescription:
        df = self.df
        if mask is not None:
            df = df[mask]

        win = df['win'].sum()
        draw = df['draw'].sum()
        lose = df.shape[0] - win - draw
        bank = df['plus'].sum()
        return StakesDescription(win, draw, lose, bank=bank)

    def plot(self, mask=None, save_name=None):
        import plotly.graph_objs as go

        df = self.df.copy()
        if mask is not None:
            df = df[mask]

        df = df.sort_values('date_time')
        x = df['date_time']
        y = df['plus'].cumsum()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='blue', width=0.5)))
        fig.update_layout(
            title='Bank line',
            xaxis_title='date',
            yaxis_title='Bank',
        )

        if save_name is None:
            fig.show()
        else:
            from paklib.io import make_dir_for_file
            make_dir_for_file(save_name)
            fig.write_image(save_name)
