from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def scatter3D(x: Tuple[str, np.ndarray], y: Tuple[str, np.ndarray], z: Tuple[str, np.ndarray], c=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[1], y[1], z[1], c=c)
    ax.set_xlabel(x[0])
    ax.set_ylabel(y[0])
    ax.set_zlabel(z[0])
    plt.show()


def animator(x, ys, method='plot', **kwargs):
    import matplotlib.animation as animation
    x = np.array(x)
    ys = np.array(ys)
    xdt = np.max(x) - np.min(x)
    ydt = np.max(ys) - np.min(ys)
    xlim = (np.min(x) - xdt * 0.1, np.max(x) + xdt * 0.1)
    ylim = (np.min(ys) - ydt * 0.1, np.max(ys) + ydt * 0.1)
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=xlim, ylim=ylim)
    if method == 'plot':
        line, = ax.plot(x, ys[0], **kwargs)
    elif method == 'scatter':
        line, = ax.scatter(x, ys[0], **kwargs)
    elif method == 'hist':
        line, = ax.hist(x, ys[0], **kwargs)
    else:
        raise StopIteration

    def init():
        line.set_data(x, ys[0])
        return line

    def animate(i):
        line.set_data(x, ys[i])
        return line

    ani = animation.FuncAnimation(fig, animate, frames=len(ys), init_func=init, interval=25, repeat=True)
    plt.show()


def histogram(pred, plot=False, xrange=None):
    values = list(pred)
    if xrange is None:
        x = sorted(set(values))
    else:
        x = list(xrange)
    y = []
    for i in x:
        y.append(values.count(i))
    al = sum(y)
    y = [i / al for i in y]
    if plot:
        plt.grid(True)
        plt.plot(x, y)
        plt.show()
    else:
        return x, y


def plot_pred(pred, ydata, key=None, min_pred=3):
    if key is not None:
        pred = pred[key]
        ydata = ydata[key]

    minim, maxim = pred.min(), pred.max()
    part = (maxim - minim) / 100

    x, y = [], []
    for i in np.linspace(minim + part, maxim - part, 100):
        if ydata[(pred >= i) & (ydata != 0.5)].size < min_pred:
            continue
        x.append(i)
        y.append(ydata[(pred >= i) & (ydata != 0.5)].mean())
    plt.plot(x, y, linewidth=1.0, label='over')

    x, y = [], []
    for i in np.linspace(minim + part, maxim - part, 100):
        if ydata[(pred <= i) & (ydata != 0.5)].size < min_pred:
            continue
        x.append(i)
        y.append((1 - ydata[(pred <= i) & (ydata != 0.5)]).mean())
    plt.plot(x, y, linewidth=1.0, label='under')

    x, y = [], []
    for i in np.linspace(minim + part, maxim - part, 100):
        if ydata[(pred >= i) & (ydata != 0.5) & (pred < i + part)].size < min_pred:
            continue
        x.append(i)
        y.append(ydata[(pred >= i) & (ydata != 0.5) & (pred < i + part)].mean())
    plt.plot(x, y, linewidth=0.3, label='in_place')

    plt.legend()
    plt.show()


def save_figure(name):
    from .io import path, make_dir_for_file
    name = path(name)
    make_dir_for_file(name)
    plt.savefig(name, dpi=200)


def ROC_curve(pred, y, show=True):
    from sklearn import metrics
    fpr, tpr, th = metrics.roc_curve(y, pred)
    score = metrics.roc_auc_score(y, pred)
    plt.plot(fpr, tpr, 'b-')
    plt.plot([0, 1], [0, 1], 'g-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve ({:.3f})'.format(score))
    if show:
        plt.show()


def _getSubplot(sub):
    if type(sub) == int:
        ax = plt.subplot(sub)
    else:
        ax = plt.subplot(*sub)
    return ax


def get_subplot_limit(subplots=None, xlim=None, ylim=None):
    if subplots is None:
        xlim = plt.xlim()
        ylim = plt.ylim()
        return xlim, ylim
    xlim = [xlim] if xlim else []
    ylim = [ylim] if ylim else []
    for sub in subplots:
        ax = _getSubplot(sub)
        xlim.append(ax.get_xlim())
        ylim.append(ax.get_ylim())
    xlim = np.array(xlim)
    ylim = np.array(ylim)
    xlim = [xlim[:, 0].min(), xlim[:, 1].max()]
    ylim = [ylim[:, 0].min(), ylim[:, 1].max()]
    if xlim[0] == xlim[1]:
        xlim = [xlim[0] - 0.1, xlim[1] + 0.1]
    if ylim[0] == ylim[1]:
        ylim = [ylim[0] - 0.1, ylim[1] + 0.1]
    return xlim, ylim


def set_subplot_limit(subplots=None, xlim=None, ylim=None):
    if subplots is None:
        plt.xlim(xlim)
        plt.ylim(ylim)
        return
    for sub in subplots:
        ax = _getSubplot(sub)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)


def get_and_set_subplot_limit(subplots=None, xlim=None, ylim=None):
    xlim, ylim = get_subplot_limit(subplots=subplots, xlim=xlim, ylim=ylim)
    set_subplot_limit(subplots=subplots, xlim=xlim, ylim=ylim)


def yield_subplot(subplots):
    for sub in subplots:
        yield _getSubplot(sub)


def action_for_subplot(subplots, action, *args, **kwargs):
    for _ in yield_subplot(subplots):
        action(*args, **kwargs)


def _get_plot_rgba(x, alpha, color):
    if alpha is None:
        a = np.full(len(x), 1.0)
    elif type(alpha) == list:
        a = alpha
    elif type(alpha) == float:
        a = np.full(len(x), alpha)
    elif alpha == 'up':
        a = np.linspace(0.1, 1.0, len(x))
    elif alpha == 'down':
        a = np.linspace(0.1, 1.0, len(x))[::-1]
    else:
        raise StopIteration(alpha)

    if color is None:
        rgb = np.full((len(x), 3), [0.0, 0.0, 0.0])
    elif type(color) == list and type(color[0]) == list:
        rgb = np.array(color)
    elif type(color) == list:
        rgb = np.full((len(x), 3), color)
    else:
        raise StopIteration(color)

    rgba = np.c_[rgb, a]
    return rgba


def scatter(x, y, key=None, show=True, color=None, alpha=None, **kwargs):
    if key is not None:
        x = x[key]
        y = y[key]
    rgba = _get_plot_rgba(x, alpha, color)
    plt.scatter(x, y, color=rgba, **kwargs)
    if show:
        plt.show()


class AlignSubplotEnd(object):
    def __init__(self, subplot=None):
        if subplot is None:
            self.subplot = plt.gca()
        else:
            self.subplot = _getSubplot(subplot)
        self.data = []

    def plot(self, y, *args, **kwargs):
        self.data.append([y, args, kwargs])

    def align(self, labeled=False):
        x_length = max([len(i[0]) for i in self.data])
        xrange = list(range(x_length))
        for y, args, kwargs in self.data:
            x = xrange[-len(y):]
            self.subplot.plot(x, y, *args, **kwargs)
        if labeled:
            self.subplot.legend()
