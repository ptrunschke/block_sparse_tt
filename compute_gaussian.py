import os
from math import comb, factorial
from itertools import product

import numpy as np
from numpy.polynomial.hermite_e import hermegauss, hermeval
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

from misc import random_homogenous_polynomial_v2, max_group_size, random_full, recover_ml, legendre_measures  #, hermite_measures
from als import ALS


# N_JOBS = -1
N_JOBS = 50


numSteps = 200
numTrials = 200
# numSteps = 20
# numTrials = 20
maxSweeps = 2000
maxIter = 100
# maxSweeps = 200
# maxIter = 10

order = 6
maxGroupSize = 1  #NOTE: This is the block size needed to represent an arbitrary polynomial.

MODE = ["TEST", "COMPUTE", "PLOT"][1]

degree = 7
f = lambda xs: np.exp(-np.linalg.norm(xs, axis=1)**2)  #NOTE: This functions gets peakier for larger M!
sampleSizes = np.unique(np.geomspace(1e1, 1e6, numSteps).astype(int))
testSampleSize = int(1e6)
# sampleSizes = np.unique(np.geomspace(1e1, 1e5, numSteps).astype(int))
# testSampleSize = int(1e5)


# def gramian(n):
#     xs, ws = hermegauss(n)
#     factors = 1/np.sqrt(np.sqrt(2*np.pi)*np.array([factorial(k) for k in range(n)]))
#     ret = np.empty((n,n))
#     for i in range(n):
#         for j in range(i+1):
#             ret[i,j] = ret[j,i] = ws @ (factors[i]*factors[j]*hermeval(xs, [0]*i+[1]) * hermeval(xs, [0]*j+[1]))
#     return ret
# assert np.linalg.norm(np.eye(10) - gramian(10)) < 1e-12


def coefficients(f, k, n):
    xs, ws = hermegauss(n)
    factors = 1/np.sqrt(np.sqrt(2*np.pi)*np.array([factorial(l) for l in range(k)]))
    ret = np.empty(k)
    for l in range(k):
        ret[l] = ws @ (factors[l] * hermeval(xs, [0]*l+[1]) * f(xs))
    return ret


exp = lambda x: np.exp(-x**2)
diff = lambda k, n: np.linalg.norm(coefficients(exp, k, n) - coefficients(exp, k, n+1))
nodes = degree+1
while diff(degree, nodes) > 1e-14:
    nodes += 1
factors = 1/np.sqrt(np.sqrt(2*np.pi)*np.array([factorial(l) for l in range(degree)]))
coefs = coefficients(exp, degree, nodes) * factors

def f_approx(xs):
    return np.product(hermeval(xs, coefs), axis=1)

# xs = np.random.randn(testSampleSize, order)
# print(np.linalg.norm(f(xs) - f_approx(xs))/np.linalg.norm(f(xs)))


def sparse_dofs():
    return comb(degree+order, order)  # all polynomials of degree at most `degree`

def bstt_dofs():
    bstts = [random_homogenous_polynomial_v2([degree]*order, deg, maxGroupSize) for deg in range(degree+1)]
    return sum(bstt.dofs() for bstt in bstts)

minimal_ranks = [1]*(order-1)
def tt_dofs_minimal_ranks():
    bstt = random_full([degree]*order, minimal_ranks)
    return bstt.dofs()

ranks = random_homogenous_polynomial_v2([degree]*order, degree, maxGroupSize).ranks
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
dof_table.add_column("sparse", justify="right")
dof_table.add_column("BSTT", justify="right")
dof_table.add_column("TT (rank 1)", justify="right")
dof_table.add_column(f"TT (rank {max(ranks)})", justify="right")
dof_table.add_column("dense", justify="right")
dof_table.add_row(f"{sparse_dofs()}", f"{bstt_dofs()}", f"{tt_dofs_minimal_ranks()}", f"{tt_dofs()}", f"{dense_dofs()}")
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


# test_points = np.random.randn(testSampleSize,order)
# test_measures = hermite_measures(test_points, degree)
#NOTE: Reconstruction w.r.t. a Gaussian measure can not work without reweighting!
test_points = 2*np.random.rand(testSampleSize,order)-1
test_measures = legendre_measures(test_points, degree)
test_values = f(test_points)
test_values_approx = f_approx(test_points)


def residual(_bstts):
    value = sum(bstt.evaluate(test_measures) for bstt in _bstts)
    return np.linalg.norm(value -  test_values) / np.linalg.norm(test_values), np.linalg.norm(value -  test_values_approx) / np.linalg.norm(test_values_approx)


def sparse_error(N):
    # points = np.random.randn(N,order)
    # measures = hermite_measures(points, degree)
    #NOTE: Reconstruction w.r.t. a Gaussian measure can not work without reweighting!
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
    value = measurement_matrix @ coefs
    return np.linalg.norm(value -  test_values) / np.linalg.norm(test_values), np.linalg.norm(value -  test_values_approx) / np.linalg.norm(test_values_approx)


def bstt_error(N, _verbosity=0):
    points = 2*np.random.rand(N,order)-1
    measures = legendre_measures(points, degree)
    values = f(points)
    return residual(recover_ml(measures, values, degree, maxGroupSize, _maxIter=maxIter, _maxSweeps=maxSweeps, _targetResidual=1e-16, _verbosity=_verbosity))


def tt_error_minimal_ranks(N):
    bstt = random_full([degree]*order, minimal_ranks)
    points = 2*np.random.rand(N,order)-1
    measures = legendre_measures(points, degree)
    values = f(points)
    solver = ALS(bstt, measures, values)
    solver.maxSweeps = maxSweeps
    solver.targetResidual = 1e-16
    solver.run()
    return residual([solver.bstt])


def tt_error(N):
    bstt = random_full([degree]*order, ranks)
    points = 2*np.random.rand(N,order)-1
    measures = legendre_measures(points, degree)
    values = f(points)
    solver = ALS(bstt, measures, values)
    solver.maxSweeps = maxSweeps
    solver.targetResidual = 1e-16
    solver.run()
    return residual([solver.bstt])


cacheDir = f".cache/gaussian_M{order}G{maxGroupSize}"
os.makedirs(cacheDir, exist_ok=True)
def compute(_error, _sampleSizes, _numTrials, _kind='exact'):
    assert _kind in ('exact', 'approx')
    cacheFile = f'{cacheDir}/{_error.__name__}.npz'
    try:
        z = np.load(cacheFile)
        assert z[f'errors_{_kind}'].shape == (_numTrials, len(_sampleSizes)) and np.all(z['sampleSizes'] == _sampleSizes)
        return z[f'errors_{_kind}']
    except:
        errors = np.empty((len(_sampleSizes), _numTrials, 2))
        for j,sampleSize in tqdm(enumerate(_sampleSizes), desc=_error.__name__, total=len(_sampleSizes)):
            errors[j] = Parallel(n_jobs=N_JOBS)(delayed(_error)(sampleSize) for _ in range(_numTrials))
        errors = errors.T
        np.savez_compressed(cacheFile, errors_exact=errors[0], errors_approx=errors[1], sampleSizes=_sampleSizes)
        return errors[int(_kind=='approx')]


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
    kind = ['exact', 'approx'][0]
    sparse_errors           = compute(sparse_error, sampleSizes, numTrials, kind)
    bstt_errors             = compute(bstt_error, sampleSizes, numTrials, kind)
    tt_errors_minimal_ranks = compute(tt_error_minimal_ranks, sampleSizes, numTrials, kind)
    # tt_errors               = compute(tt_error, sampleSizes, numTrials, kind)
if MODE == "PLOT":
    # def plot(xs, ys1, ys2, ys3, ys4):
    def plot(xs, ys1, ys2, ys3):
        fontsize = 10

        # BG = coloring.bimosyellow
        BG = "xkcd:white"
        C0 = "C0"
        C1 = "C1"
        C2 = "C2"
        C3 = "C3"
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
        # plotting.plot_quantiles(xs, ys4, qrange=(0.15,0.85), num_quantiles=5, linewidth_fan=1, color=C3, axes=ax, zorder=2)

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
        plt.savefig(f"figures/gaussian.png", dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches="tight") # , transparent=True)

    plot(sampleSizes, sparse_errors, bstt_errors, tt_errors_minimal_ranks)
    # plot(sampleSizes, sparse_errors, bstt_errors, tt_errors_minimal_ranks, tt_errors)