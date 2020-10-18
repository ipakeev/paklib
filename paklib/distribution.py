from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import poisson, nbinom, norm, chisquare

from .plt import histogram


def is_equal(sc1, sc2):
    sc1, sc2 = [np.asarray(i, np.int32) for i in [sc1, sc2]]
    x1, y1 = histogram(sc1)
    x2, y2 = histogram(sc2)
    x1, x2, y1, y2 = [np.array(i) for i in [x1, x2, y1, y2]]
    x_all = sorted(set(list(x1) + list(x2)))
    p1, p2 = [], []
    for n in x_all:
        if not x1[x1 == n].size or not x2[x2 == n].size:
            continue
        if y1[x1 == n][0] == 0.0 or y2[x2 == n][0] == 0.0:
            continue
        p1.append(y1[x1 == n][0])
        p2.append(y2[x2 == n][0])
    chi = chisquare(p1, p2)
    return chi


def find_best_similar(scores, mu_poisson=0.0, mu_nbinom=0.0, corr_nbinom=0.5, mu_norm=0.0, corr_norm=0.0):
    mb = np.array(scores, np.int32)
    xx, yy = histogram(scores)
    xx, yy = [np.array(i) for i in [xx, yy]]
    zz = poisson.pmf(xx, mb.mean() + mu_poisson)
    chi1 = chisquare(yy, zz)[0]
    zz = nbinom.pmf(xx, mb.mean() + mu_nbinom, corr_nbinom)
    chi2 = chisquare(yy, zz)[0]
    zz = norm.pdf(xx, mb.mean() + mu_norm, mb.std() + corr_norm)
    chi3 = chisquare(yy, zz)[0]
    random = ['poisson', 'nbinom', 'norm']
    chi = [chi1, chi2, chi3]
    best = random[chi.index(min(chi))]
    print('   Best: {}'.format(best))
    print('Poisson: {:.3f}'.format(chi[0]))
    print('Nbinom : {:.3f}'.format(chi[1]))
    print('Norm   : {:.3f}'.format(chi[2]))


def visualize(scores, xrange=None, mu_poisson=0.0, mu_nbinom=0.0, corr_nbinom=0.5, mu_norm=0.0, corr_norm=0.0):
    if xrange is None:
        xrange = np.arange(30)
    else:
        xrange = np.array(list(xrange))
    scores = np.asrray(scores)

    x, y = histogram(scores)
    plt.grid(True)
    plt.plot(x, y, 'r-', label='current', linewidth=0.5)

    y = poisson.pmf(xrange, scores.mean() + mu_poisson)
    plt.plot(xrange, y, 'y-', label='poisson')

    y = nbinom.pmf(xrange, scores.mean() + mu_nbinom, corr_nbinom)
    plt.plot(xrange, y, 'g-', label='nbinom')

    y = norm.pdf(xrange, scores.mean() + mu_norm, scores.std() + corr_norm)
    plt.plot(xrange, y, 'b-', label='norm')

    plt.legend(loc='best')
    plt.show()


def best_poisson(scores, mu_lim: Tuple[float, float] = None):
    def func(x):
        if not (x[0] >= 0.5):
            return 10000
        y = poisson.pmf(xrange, x[0])
        chi = chisquare(dist, y)[0]
        if np.isnan(chi) or np.isinf(chi):
            return 10000
        return chi

    if mu_lim is None:
        muLimMin, muLimMax = -1.0, 1.0
    else:
        muLimMin, muLimMax = mu_lim

    scores = np.array(scores)
    xrange, dist = histogram(scores)
    bounds = [(scores.mean() + muLimMin, scores.mean() + muLimMax)]

    de = differential_evolution(func, bounds)
    best = [de.x[0] - scores.mean()]
    find_best_similar(scores)
    print('\nBounds: [{:.3f}]'.format(*best))
    find_best_similar(scores, mu_poisson=best[0])
    return best


def best_nbinom(scores, mu_lim: Tuple[float, float] = None, corr_lim: Tuple[float, float] = None):
    def func(x):
        if not (x[0] >= 0.5 and 0.0 < x[1] < 1.0):
            return 10000
        y = nbinom.pmf(xrange, x[0], x[1])
        chi = chisquare(dist, y)[0]
        if np.isnan(chi) or np.isinf(chi):
            return 10000
        return chi

    if mu_lim is None:
        muLimMin, muLimMax = -2.0, 2.0
    else:
        muLimMin, muLimMax = mu_lim
    if corr_lim is None:
        corrLim = (0.4, 0.6)
    else:
        corrLim = corr_lim

    scores = np.array(scores)
    xrange, dist = histogram(scores)
    bounds = [(scores.mean() + muLimMin, scores.mean() + muLimMax), corrLim]

    de = differential_evolution(func, bounds)
    best = [de.x[0] - scores.mean(), de.x[1]]
    find_best_similar(scores)
    print('\nBounds: [{:.3f}, {:.3f}]'.format(*best))
    find_best_similar(scores, mu_nbinom=best[0], corr_nbinom=best[1])
    return best


def best_norm(scores, mu_lim: Tuple[float, float] = None, corr_lim: Tuple[float, float] = None):
    def func(x):
        if not (x[0] >= 0.5 and -20 < x[1] < 20):
            return 10000
        y = norm.pdf(xrange, x[0], x[1])
        chi = chisquare(dist, y)[0]
        if np.isnan(chi) or np.isinf(chi):
            return 10000
        return chi

    if mu_lim is None:
        muLimMin, muLimMax = -5.0, 5.0
    else:
        muLimMin, muLimMax = mu_lim
    if corr_lim is None:
        corrLimMin, corrLimMax = -5.0, 5.0
    else:
        corrLimMin, corrLimMax = corr_lim

    scores = np.array(scores)
    xrange, dist = histogram(scores)
    bounds = [(scores.mean() + muLimMin, scores.mean() + muLimMax),
              (scores.std() + corrLimMin, scores.std() + corrLimMax)]

    de = differential_evolution(func, bounds)
    best = [de.x[0] - scores.mean(), de.x[1] - scores.std()]
    find_best_similar(scores)
    print('\nBounds: [{:.3f}, {:.3f}]'.format(*best))
    find_best_similar(scores, mu_norm=best[0], corr_norm=best[1])
    return best


def proba_norm(mu, std, xrange, bk):
    xrange = np.array(list(xrange))
    p = norm.pdf(xrange, mu, std)
    over = p[xrange > bk].sum()
    under = p[xrange < bk].sum()
    al = over + under
    over, under = over / al, under / al
    over, under = round(over, 3), round(under, 3)
    return over, under
