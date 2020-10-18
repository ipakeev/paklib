import numpy as np


def get_payout(odd):
    odd = np.asarray(odd)
    return np.around(1 / np.sum(1 / odd), 3)


def get_proba(odd):
    odd = np.asarray(odd)
    payout = get_payout(odd)
    return np.around(payout / odd, 4)


def get_odd(proba, payout=0.9):
    proba = np.clip(proba, 0.01, 0.99)
    odd = payout / proba
    odd = np.clip(odd, 1.01, 10.0)
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


def exp_probability(r1, r2, alpha=0.0, beta=1.0, kLine=1.0, bLine=0.0):
    p = np.exp(alpha + beta * (r1 - r2)) / (1 + np.exp(alpha + beta * (r1 - r2)))
    p = kLine * p + bLine
    return p


def elo(r1, r2, real_points, g=None):
    return 20.0 * (real_points - probability(r1, r2))


def elo_goal(r1, r2, real_points, g):
    dg = 1.0 if g <= 1 else 1.5 if g == 2 else ((11.0 + g) / 8)
    return 20.0 * dg * (real_points - probability(r1, r2))


def elo_value(r1, r2, real_points, g):
    dg = g * 10
    return 20.0 * dg * (real_points - probability(r1, r2))


def DNB(odd, proba=False):
    p = get_proba(odd)
    if len(odd) == 2:
        return list(p) if proba else list(odd)
    al = p[0] + p[-1]
    p = [p[0] / al, p[2] / al]
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
    p = [p[0] + p[1], p[1] + p[2], p[0] + p[2]]
    if proba:
        return p
    return list(get_odd(p))


def DNB_DC(odd, proba=False):
    p = get_proba(odd)
    if len(odd) == 2:
        if proba:
            return list(p) + list(p) + [1.0]
        else:
            return list(odd) + list(odd) + [1.0]
    al = p[0] + p[-1]
    p = [p[0] / al, p[2] / al, p[0] + p[1], p[1] + p[2], p[0] + p[2]]
    if proba:
        return p
    return list(get_odd(p))


def kelly(pred, odd, key=None):
    if key is not None:
        pred = pred[key]
        odd = odd[key]
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
