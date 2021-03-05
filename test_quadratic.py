import os

import numpy as np
from numpy.polynomial.legendre import legval
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D

from misc import random_polynomial, random_full
from als import ALS

import coloring
import plotting


# ==========
#  TRAINING
# ==========

num_trials = 20
order = 6
degree = 6
max_degree = 8
test_samples = int(1e4)

Ns = np.unique(np.geomspace(10, 1e4, 20).astype(int))
f = lambda xs: np.linalg.norm(xs, axis=1)**2


def draw_samples(N):
    samples = 2*np.random.rand(N,order)-1
    factors = np.sqrt(2*np.arange(degree+1)+1)
    measures = legval(samples, np.diag(factors)).T
    assert measures.shape == (order,N,degree+1)
    values = f(samples)
    assert values.shape == (N,)
    return measures, values


def sparse_error(N, maxIter):
    bstt = random_polynomial([degree]*order, max_degree)
    solver = ALS(bstt, *draw_samples(N))
    solver.maxSweeps = maxIter
    solver.minDecrease = 1e-4
    solver.targetResidual = 1e-16
    solver.run()
    return ALS(solver.bstt, *draw_samples(test_samples)).residual()


def dense_error(N, maxIter):
    bstt = random_full([degree]*order, max_degree+1)
    solver = ALS(bstt, *draw_samples(N))
    solver.maxSweeps = maxIter
    solver.minDecrease = 1e-4
    solver.targetResidual = 1e-16
    solver.run()
    return ALS(solver.bstt, *draw_samples(test_samples)).residual()


def compute(Ns, error, maxIter=100):
    os.makedirs(".cache", exist_ok=True)
    err_file = f'.cache/{error.__name__}.npz'
    try:
        z = np.load(err_file)
        errors = z['errors']
    except:
        errors = np.empty((len(Ns), num_trials))
        for j, N in enumerate(tqdm(Ns)):
            for k in trange(num_trials):
                errors[j,k] = error(N, maxIter)
        np.savez_compressed(err_file, errors=errors)
    return errors


def plot(xs, ys1, ys2):
    fontsize = 10

    # BG = coloring.bimosyellow
    BG = "xkcd:white"
    C0 = coloring.mix(coloring.bimosred, 80)
    C1 = "xkcd:black"
    # C2 = "xkcd:dark grey"
    # C3 = "xkcd:grey"

    geometry = {
        'top':    1,
        'bottom': 0,
        'left':   0,
        'right':  1,
        'wspace': 0.25,  # the default as defined in rcParams
        'hspace': 0.25  # the default as defined in rcParams
    }
    figshape = (1,1)
    figsize = coloring.compute_figsize(geometry, figshape, 2)
    fig,ax = plt.subplots(*figshape, figsize=figsize, dpi=300)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    plotting.plot_quantiles(xs, ys1, qrange=(0.15,0.85), num_quantiles=5, linewidth_fan=1, color=C0, axes=ax, zorder=2)
    plotting.plot_quantiles(xs, ys2, qrange=(0.15,0.85), num_quantiles=5, linewidth_fan=1, color=C1, axes=ax, zorder=2)

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(xs[0], xs[-1])
    ax.set_yticks(10.0**np.arange(-16,7), minor=True)
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_yticks(10.0**np.arange(-16,7,3))
    ax.set_xlabel(r"$n$", fontsize=fontsize)
    ax.set_ylabel(r"rel. error", fontsize=fontsize)

    legend_elements = [
        Line2D([0], [0], color=coloring.mix(C0, 80), lw=1.5),  #NOTE: stacking multiple patches seems to be hard. This is the way seaborn displays such graphs
        Line2D([0], [0], color=coloring.mix(C1, 80), lw=1.5),  #NOTE: stacking multiple patches seems to be hard. This is the way seaborn displays such graphs
    ]
    ax.legend(legend_elements, ["sparse", "dense"], loc='lower left', fontsize=fontsize)

    plt.subplots_adjust(**geometry)
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/sparse_vs_dense.png", dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches="tight") # , transparent=True)


sparse_errors = compute(Ns, sparse_error).T
dense_errors = compute(Ns, dense_error).T
plot(Ns, sparse_errors, dense_errors)
