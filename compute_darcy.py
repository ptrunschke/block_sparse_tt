import os
from math import comb, factorial
from itertools import product

import numpy as np
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from joblib import Parallel, delayed

from misc import random_homogenous_polynomial_v2, max_group_size, random_full, recover_ml, legendre_measures  #, hermite_measures
from als import ALS


N_JOBS = -1
CACHE_DIRECTORY = ".cache/darcy_M{order}G{maxGroupSize}"
DATAFILE = ".cache/darcy_uniform_integral.npz"


# numSteps = 20
# numTrials = 20
# maxSweeps = 200
# maxIter = 10
numSteps = 200
numTrials = 200
maxSweeps = 2000
maxIter = 100

degree = 5
maxGroupSize = 3  #NOTE: This is the block size needed to represent an arbitrary polynomial.

z = np.load(DATAFILE)
maxSampleSize, order = z['samples'].shape
assert maxSampleSize/2 > 1e1
assert z['values'].shape == (maxSampleSize,)
sampleSizes = np.unique(np.geomspace(1e1, maxSampleSize/2, numSteps).astype(int))
testSampleSize = maxSampleSize-sampleSizes[-1]
CACHE_DIRECTORY = CACHE_DIRECTORY.format(order=order, maxGroupSize=maxGroupSize)

all_points = z['samples']
all_values = z['values']
test_points = all_points[sampleSizes[-1]:]
assert len(test_points) == testSampleSize
test_measures = legendre_measures(test_points, degree)
test_values = all_values[sampleSizes[-1]:]


def sparse_dofs():
    return comb(degree+order, order)  # all polynomials of degree at most `degree`

def bstt_dofs():
    bstts = [random_homogenous_polynomial_v2([degree]*order, deg, maxGroupSize) for deg in range(degree+1)]
    return sum(bstt.dofs() for bstt in bstts)

ranks = random_homogenous_polynomial_v2([degree]*order, degree, maxGroupSize).ranks
def tt_dofs():
    bstt = random_full([degree]*order, ranks)
    return bstt.dofs()

def dense_dofs():
    return (degree+1)**order


console = Console()

console.rule(f"[bold]Darcy")

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
dof_table.add_column(f"TT (rank {max(ranks)})", justify="right")
dof_table.add_column("dense", justify="right")
dof_table.add_row(f"{sparse_dofs()}", f"{bstt_dofs()}", f"{tt_dofs()}", f"{dense_dofs()}")
console.print()
console.print(dof_table, justify="center")


def multiIndices(_degree, _order):
    return filter(lambda mI: sum(mI) <= _degree, product(range(_degree+1), repeat=_order))  # all polynomials of degree at most `degree`


def residual(_bstts):
    value = sum(bstt.evaluate(test_measures) for bstt in _bstts)
    return np.linalg.norm(value -  test_values) / np.linalg.norm(test_values)


def sparse_error(N):
    assert N <= sampleSizes[-1]
    points = all_points[:N]
    measures = legendre_measures(points, degree)
    values = all_values[:N]
    j = np.arange(order)
    measurement_matrix = np.empty((N,sparse_dofs()))
    for e,mIdx in enumerate(multiIndices(degree, order)):
        measurement_matrix[:,e] = np.product(measures[j, :, mIdx], axis=0)
    coefs, *_ = np.linalg.lstsq(measurement_matrix, values, rcond=None)
    measurement_matrix = np.empty((testSampleSize,sparse_dofs()))
    for e,mIdx in enumerate(multiIndices(degree, order)):
        measurement_matrix[:,e] = np.product(test_measures[j, :, mIdx], axis=0)
    value = measurement_matrix @ coefs
    return np.linalg.norm(value -  test_values) / np.linalg.norm(test_values)


def bstt_error(N, _verbosity=0):
    assert N <= sampleSizes[-1]
    points = all_points[:N]
    measures = legendre_measures(points, degree)
    values = all_values[:N]
    return residual(recover_ml(measures, values, degree, maxGroupSize, _maxIter=maxIter, _maxSweeps=maxSweeps, _targetResidual=1e-16, _verbosity=_verbosity))


def tt_error(N):
    assert N <= sampleSizes[-1]
    bstt = random_full([degree]*order, ranks)
    points = all_points[:N]
    measures = legendre_measures(points, degree)
    values = all_values[:N]
    solver = ALS(bstt, measures, values)
    solver.maxSweeps = maxSweeps
    solver.targetResidual = 1e-16
    solver.run()
    return residual([solver.bstt])


os.makedirs(CACHE_DIRECTORY, exist_ok=True)
def compute(_error, _sampleSizes, _numTrials):
    cacheFile = f'{CACHE_DIRECTORY}/{_error.__name__}.npz'
    try:
        z = np.load(cacheFile)
        if z['errors'].shape != (_numTrials, len(_sampleSizes)):
            print(f"WARNING: errors.shape={z['errors'].shape} != {(_numTrials, len(_sampleSizes))}")
        if np.any(z['sampleSizes'] != _sampleSizes):
            print(f"WARNING: sampleSizes != _sampleSizes")
        errors = z['errors']
    except:
        errors = np.full((_numTrials, len(_sampleSizes)), np.nan)
    assert errors.shape == (_numTrials, len(_sampleSizes))
    # Go through all sampleSizes and recompute those errors that contain np.nan
    for j,sampleSize in tqdm(enumerate(_sampleSizes), desc=_error.__name__, total=len(_sampleSizes)):
        if np.any(np.isnan(errors[:,j])):
            if not np.all(np.isnan(errors[:,j])):
                print("WARNING: Only {np.count_nonzero(np.isnan(errors[:,j]))} errors are NaN.")
            errors[:,j] = Parallel(n_jobs=N_JOBS)(delayed(_error)(sampleSize) for _ in range(_numTrials))
            np.savez_compressed(cacheFile, errors=errors, sampleSizes=_sampleSizes)


if __name__ == "__main__":
    console.print()
    compute(sparse_error, sampleSizes, numTrials)
    compute(bstt_error, sampleSizes, numTrials)
    compute(tt_error, sampleSizes, numTrials)
