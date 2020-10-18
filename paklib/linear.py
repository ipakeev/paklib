import matplotlib.pylab as plt
import numpy as np
from scipy import stats
from scipy.optimize import differential_evolution


def list_window(x, window, iterations=1, func=np.median):
    x = [func(x[i:i + window]) for i in range(len(x) - window)]
    if iterations == 1:
        return np.array(x)
    return list_window(x, window, iterations - 1)


def MA(x=None, y=None, n=2):
    if x is None:
        x = np.arange(len(y))
    else:
        x = np.array(x)
    y = np.array(y)
    p = np.polyfit(x, y, n)
    yp = np.polyval(p, x)
    return yp


def MA_mean(x=None, y=None, n=2):
    if x is None:
        x = np.arange(len(y))
    else:
        x = np.array(x)
    y = np.array(y)

    y[0] = y[:3].mean()
    y[-1] = y[-3:].mean()

    p = np.polyfit(x, y, n)
    yp = np.polyval(p, x)
    return yp


def KB(x1, y1, x2, y2):
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    return k, b


def get_outlier_limits(y, k_drop=1.5):
    p25 = np.percentile(y, 25)
    p75 = np.percentile(y, 75)
    lower = p25 - k_drop * (p75 - p25)
    upper = p75 + k_drop * (p75 - p25)
    return lower, upper


def drop_linear_outlier(x, y, k_drop=1.5, return_key=False):
    x = np.array(x, dtype=np.float)
    y = np.array(y, dtype=np.float)

    k, b, *_ = stats.linregress(x, y)
    k = 0.0001 if k == 0.0 else k

    yMust = k * x + b
    A = [x, yMust]
    B = [(y - b) / k, y]
    C = [x, y]
    AB = np.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)
    BC = abs(C[0] - B[0])
    AC = C[1] - A[1]
    sin = BC[0] / AB[0]
    h = AC * sin

    lower, upper = get_outlier_limits(h, k_drop=k_drop)
    key = np.logical_and(h >= lower, h <= upper)
    if return_key:
        return key

    return x[key], y[key]


def polynomial(x, y, degree=4):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    x = x[:, np.newaxis]
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(x, y)
    y = pipeline.predict(x)
    return x.ravel(), y


def sinus(gene, x, bk):
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(bk, list):
        bk = np.array(bk)
    a, b, c, m, n = gene
    return a * np.sin((x + b) / c) + m * x + n + bk


def sinus_speed(gene, x, bk=None):
    if isinstance(x, list):
        x = np.array(x)
    a, b, c, m, n = gene
    v = a * np.cos((b + x) / c) / c + m
    return v


def sinus_acceleration(gene, x, bk=None):
    if isinstance(x, list):
        x = np.array(x)
    a, b, c, m, n = gene
    v = -a * np.sin((b + x) / c) / (c * c)
    return v


class GeneticSinus(object):
    def __init__(self, x, y, bk, key=None):
        self.x = np.array(x)
        self.y = np.array(y)
        self.bk = np.array(bk)
        if key is not None:
            key = np.array(key)
            self.x = self.x[key == True]
            self.y = self.y[key == True]
            self.bk = self.bk[key == True]

        mx = np.max(self.y) - np.min(self.y)
        self.bounds = [(mx / 8.0, mx / 2.0), (0.0, 50.0), (20.0, 100.0), (-0.1, 0.1), (-20.0, 20.0)]
        self.gene = None
        self._model = None
        self._fig = None

    @staticmethod
    def result(gene, x, y, bk):
        pred = sinus(gene, x, bk)
        value1 = 1 - stats.pearsonr(pred, y)[0]
        value2 = np.absolute(pred - y).mean()
        if np.isnan(value1) or np.isnan(value2):
            return 10000
        value = value1 * value2
        return value

    def fit(self, strategy='best1bin', maxiter=200, popsize=200, mutation=(0.5, 1.0), recombination=0.7):
        self._model = differential_evolution(self.result, self.bounds, (self.x, self.y, self.bk),
                                             strategy=strategy, maxiter=maxiter, popsize=popsize,
                                             mutation=mutation, recombination=recombination)
        self.gene = self._model.x

    def predict(self, x, bk):
        assert self.gene is not None
        return sinus(self.gene, x, bk)

    def plot(self, x_now, bk_now, show=True):
        from .distribution import proba_norm

        p = self.predict(self.x, self.bk)
        pred = self.predict(x_now, bk_now)

        fig = plt.figure()

        ax = fig.add_subplot(221)
        ax.scatter(self.x, self.y, c='g')
        ax.plot(self.x, p, 'b-', linewidth=0.5)
        ax.plot([self.x[-1], x_now], [p[-1], pred], 'b--', linewidth=0.5)
        ax.plot(self.x, self.bk, 'r--', linewidth=0.2)
        ax.plot([self.x[-1], x_now], [self.bk[-1], bk_now], 'r--', linewidth=0.2)
        ax.set_title('pred = {:.2f}'.format(pred))

        ax = fig.add_subplot(222)
        ax.scatter(self.x, self.y - self.bk, c='g')
        ax.plot(self.x, p - self.bk, 'b-', linewidth=0.5)
        ax.plot([self.x[-1], x_now], [p[-1] - self.bk[-1], pred - bk_now], 'b--', linewidth=0.5)
        ax.plot([self.x[0], x_now], [0.0, 0.0], 'k--', linewidth=0.5)
        std = (self.y - p).std()
        over, under = proba_norm(pred, std, range(0, 150), bk_now)
        ax.set_title('OU = [{:.2f}, {:.2f}]'.format(over, under))

        ax = fig.add_subplot(224)
        ax.scatter(self.x, self.y - p, c='g')
        ax.plot([self.x[0], x_now], [0.0, 0.0], 'k--', linewidth=0.5)

        if show:
            plt.show()
        self._fig = fig

    def close_plot(self):
        plt.close(self._fig)

    def get_not_outlier_key(self, k_drop=1.5):
        pred = self.predict(self.x, self.bk)
        y = self.y - pred
        lower, upper = get_outlier_limits(y, k_drop=k_drop)
        key = np.array([lower <= i <= upper for i in y])
        return key
