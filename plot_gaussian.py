import os
from math import comb
from itertools import product

import numpy as np
import xerus as xe
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D
import coloring
import plotting

from misc import random_homogenous_polynomial_v2, max_group_size, legendre_measures, random_full, recover_ml
from als import ALS
from riccati import riccati_matrices


# numSteps = 200
# numTrials = 200
numSteps = 20
numTrials = 20
maxSweeps = 2000
maxIter = 100
# maxSweeps = 200
# maxIter = 10

order = 8
maxGroupSize = 1  #NOTE: This is the block size needed to represent an arbitrary polynomial.

MODE = ["TEST", "COMPUTE", "PLOT"][1]

degree = 7
f = lambda xs: np.exp(-np.linalg.norm(xs, axis=1)**2)  #NOTE: This functions gets peakier for larger M!
sampleSizes = np.unique(np.geomspace(1e1, 1e6, numSteps).astype(int))
testSampleSize = int(1e6)
# testSampleSize = int(1e4)


def sparse_dofs():
    return comb(degree+order, order)  # all polynomials of degree at most `degree`

def bstt_dofs():
    bstts = [random_homogenous_polynomial_v2([degree]*order, deg, maxGroupSize) for deg in range(degree+1)]
    return sum(bstt.dofs() for bstt in bstts)

##TODO: Is this a sensible choice?
#ranks = random_homogenous_polynomial_v2([degree]*order, degree, maxGroupSize).ranks
ranks = [1]*(order-1)
def tt_dofs():
    bstt = random_full([degree]*order, ranks)
    return bstt.dofs()

def dense_dofs():
    return (degree+1)**order


console = Console()

console.rule(f"[bold]Gaussian")

parameter_table = Table(title="Parameters", title_style="bold", show_header=False)
parameter_table.add_column(justify="left")
parameter_table.add_column(justify="right")
parameter_table.add_row("Order", f"{order}")
parameter_table.add_row("Degree", f"{degree}")
parameter_table.add_row("Maximal group size", f"{maxGroupSize}")
parameter_table.add_row("Maximal possible group size", f"{max_group_size(order, degree)}")
parameter_table.add_row("Rank", f"{max(ranks)}")
parameter_table.add_row("Number of samples", f"{sampleSizes[0]} --> {sampleSizes[-1]}")
parameter_table.add_row("Maximal number of sweeps", f"{maxSweeps}")
console.print()
console.print(parameter_table, justify="center")

dof_table = Table(title="Dofs", title_style="bold")
dof_table.add_column("sparse", justify="left")
dof_table.add_column("BSTT", justify="left")
dof_table.add_column("TT", justify="left")
dof_table.add_column("dense", justify="left")
dof_table.add_row(f"{sparse_dofs()}", f"{bstt_dofs()}", f"{tt_dofs()}", f"{dense_dofs()}")
console.print()
console.print(dof_table, justify="center")

# layout = Layout()
# layout.split(
#     Layout(parameter_table, name="left"),
#     Layout(dof_table, name="right"),
#     direction="horizontal"
# )
# console.print(layout)


def multiIndices(_degree, _order):
    return filter(lambda mI: sum(mI) <= _degree, product(range(_degree+1), repeat=_order))  # all polynomials of degree at most `degree`


test_points = 2*np.random.rand(testSampleSize,order)-1
test_measures = legendre_measures(test_points, degree)
test_values = f(test_points)


def residual(_bstts):
    bstts_value = sum(bstt.evaluate(test_measures) for bstt in _bstts)
    return np.linalg.norm(bstts_value -  test_values) / np.linalg.norm(test_values)


def sparse_error(N):
    points = 2*np.random.rand(N,order)-1
    measures = legendre_measures(points, degree)
    values = f(points)
    j = np.arange(order)
    measurement_matrix = np.empty((N,sparse_dofs()))
    for e,mIdx in enumerate(multiIndices(degree, order)):
        measurement_matrix[:,e] = np.product(measures[j, :, mIdx], axis=0)
    coefs, *_ = np.linalg.lstsq(measurement_matrix, values, rcond=None)
    measurement_matrix = np.empty((testSampleSize,sparse_dofs()))
    for e,mIdx in enumerate(multiIndices(degree, order)):
        measurement_matrix[:,e] = np.product(test_measures[j, :, mIdx], axis=0)
    return np.linalg.norm(measurement_matrix @ coefs -  test_values) / np.linalg.norm(test_values)


def bstt_error(N, _verbosity=0):
    points = 2*np.random.rand(N,order)-1
    measures = legendre_measures(points, degree)
    values = f(points)
    return residual(recover_ml(measures, values, degree, maxGroupSize, _maxIter=maxIter, _maxSweeps=maxSweeps, _targetResidual=1e-16, _verbosity=_verbosity))


def tt_error(N):
    bstt = random_full([degree]*order, degree+1)
    points = 2*np.random.rand(N,order)-1
    measures = legendre_measures(points, degree)
    values = f(points)
    solver = ALS(bstt, measures, values)
    solver.maxSweeps = maxSweeps
    solver.targetResidual = 1e-16
    solver.run()
    return residual([solver.bstt])


cacheDir = f".cache/M{order}G{maxGroupSize}"
os.makedirs(cacheDir, exist_ok=True)
def compute(_error, _sampleSizes, _numTrials):
    cacheFile = f'{cacheDir}/{_error.__name__}.npz'
    try:
        z = np.load(cacheFile)
        assert z['errors'].shape == (_numTrials, len(_sampleSizes)) and np.all(z['sampleSizes'] == _sampleSizes)
        return z['errors']
    except:
        errors = np.empty((len(_sampleSizes), _numTrials))
        for j,sampleSize in tqdm(enumerate(_sampleSizes), desc=_error.__name__, total=len(_sampleSizes)):
            errors[j] = Parallel(n_jobs=-1)(delayed(_error)(sampleSize) for _ in range(_numTrials))
        errors = errors.T
        np.savez_compressed(cacheFile, errors=errors, sampleSizes=_sampleSizes)
    return errors


if MODE == "TEST":
    error_table = Table(title="Errors", title_style="bold")
    error_table.add_column("sparse", justify="left")
    error_table.add_column("BSTT", justify="left")
    error_table.add_column("TT", justify="left")
    error_table.add_column("dense", justify="left")
    N = int(1e3)
    error_table.add_row(f"{sparse_error(N):.2e}", f"{bstt_error(N, _verbosity=1):.2e}", f"{tt_error(N):.2e}", f"")
    console.print()
    console.print(error_table, justify="center")
elif MODE in ["COMPUTE", "PLOT"]:
    console.print()
    compute(sparse_error, sampleSizes, numTrials)
    compute(bstt_error, sampleSizes, numTrials)
    compute(tt_error, sampleSizes, numTrials)
if MODE == "PLOT":
    def plot(xs, ys1, ys2, ys3):
        fontsize = 10

        # BG = coloring.bimosyellow
        BG = "xkcd:white"
        C0 = "C0"
        C1 = "C1"
        C2 = "C2"
        # C0 = coloring.mix(coloring.bimosred, 80)
        # C1 = "xkcd:black"
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
        plotting.plot_quantiles(xs, ys3, qrange=(0.15,0.85), num_quantiles=5, linewidth_fan=1, color=C2, axes=ax, zorder=2)

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(xs[0], xs[-1])
        ax.set_yticks(10.0**np.arange(-16,7), minor=True)
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_yticks(10.0**np.arange(-16,7,3))
        ax.set_xlabel(r"\# sampels", fontsize=fontsize)
        ax.set_ylabel(r"rel. error", fontsize=fontsize)

        legend_elements = [
            Line2D([0], [0], color=coloring.mix(C0, 80), lw=1.5),  #NOTE: stacking multiple patches seems to be hard. This is the way seaborn displays such graphs
            Line2D([0], [0], color=coloring.mix(C1, 80), lw=1.5),  #NOTE: stacking multiple patches seems to be hard. This is the way seaborn displays such graphs
            Line2D([0], [0], color=coloring.mix(C2, 80), lw=1.5),  #NOTE: stacking multiple patches seems to be hard. This is the way seaborn displays such graphs
        ]
        ax.legend(legend_elements, ["sparse", "block-sparse TT", "dense TT"], loc='upper right', fontsize=fontsize)

        plt.subplots_adjust(**geometry)
        os.makedirs("figures", exist_ok=True)
        plt.savefig(f"figures/sparse_vs_dense.png", dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches="tight") # , transparent=True)


    sparse_errors = compute(sparse_error, sampleSizes, numTrials)
    bstt_errors   = compute(bstt_error, sampleSizes, numTrials)
    tt_errors     = compute(tt_error, sampleSizes, numTrials)
    plot(sampleSizes, sparse_errors, bstt_errors, tt_errors)
