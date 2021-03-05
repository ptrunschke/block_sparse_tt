import numpy as np
import matplotlib.pyplot as plt


def plot_quantiles(nodes, values, qrange=(0,1), ax=None, num_quantiles=4, linewidth_fan=0, **kwargs):
    """
    Plot the quantiles for a stochastic process.

    Parameters
    ----------
    nodes : ndarray (shape: (n,))
        Nodes at which the process is measured (index set, plotted on the x-axis).
    values : ndarray (shape: (m,n))
        Realizations of the the process (plotted on the y-axis).
        Each row of values contains a different path of the stochastic process.
    qrange : float (2,)
        Quantile range to use. (Outer bounds are currently excluded.)
    ax : matplotlib.axes.Axes, optional
        The axis object used for plotting. (default: matplotlib.pyplot.gca())
    num_quantiles : int (>= 0), optional
        Number of quantiles to plot. (default: 4)
    linewidth_fan : float, optional
        Linewidth of the bounding lines of each quantile. (default: 0)
    """
    errors = values
    values = nodes
    assert values.ndim == 1
    assert len(values) == errors.shape[1]
    assert num_quantiles >= 0
    if ax is None:
        ax = plt.gca()

    if num_quantiles == 0:
        ps = np.full((1,), 0.5)
        alphas = np.full((1,), kwargs.get('alpha', 1))
    else:
        ps = np.linspace(0,1,2*(num_quantiles+1)+1)[1:-1]
        alphas = np.empty(num_quantiles)
        alphas[0] = 2*ps[0]
        for i in range(1, num_quantiles):
            alphas[i] = 2*(ps[i] - ps[i-1])/(1 - 2*ps[i-1])
        alphas *= kwargs.get('alpha', 1)

    qs = np.quantile(errors, ps, axis=0)
    zorder = kwargs.pop('zorder', 1)
    base_line, = ax.plot(values, qs[num_quantiles], zorder=zorder+num_quantiles, **kwargs)
    ls = [base_line]
    for e in range(num_quantiles):
        l = ax.fill_between(values, qs[e], qs[-1-e], color=base_line.get_color(), alpha=alphas[e], zorder=zorder+e, linewidth=linewidth_fan)
        ls.append(l)
    return ls, alphas
