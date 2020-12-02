import numpy as np


def e_score_mean(scores, double e=2.7183, double alpha=-0.0065):
    cdef double x = 0.0, y = 0.0, v
    cdef int n
    for n, i in enumerate(scores[::-1]):
        v = e ** (alpha * n)
        x += v
        y += i * v
    return y / x


def probability(double r1, double r2):
    return 1 / (1 + 10 ** ((r2 - r1) / 400))


def exp_probability(double r1, double r2, double alpha=0.0, double beta=1.0, double k=1.0, double b=0.0):
    cdef double p
    p = np.exp(alpha + beta * (r1 - r2)) / (1 + np.exp(alpha + beta * (r1 - r2)))
    return k * p + b


def elo(double r1, double r2, double real_points, g=None):
    return 20.0 * (real_points - probability(r1, r2))


def elo_goal(double r1, double r2, double real_points, double g):
    cdef double dg
    dg = 1.0 if g <= 1 else 1.5 if g == 2 else ((11.0 + g) / 8)
    return 20.0 * dg * (real_points - probability(r1, r2))


def elo_value(double r1, double r2, double real_points, double g):
    cdef double dg
    dg = g * 10
    return 20.0 * dg * (real_points - probability(r1, r2))

