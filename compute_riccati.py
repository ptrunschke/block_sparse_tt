import os
from math import comb
from itertools import product

import numpy as np
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from joblib import Parallel, delayed

from misc import random_homogenous_polynomial_v2, max_group_size, monomial_measures, random_full
from als import ALS
from riccati import riccati_matrices


#TODO: Use a change of basis given by the diagonalization of the Riccati equation --> low rank. (Mit Leon reden!)


N_JOBS = -1
CACHE_DIRECTORY = ".cache/riccati_M{order}G{maxGroupSize}"


numSteps = 200
numTrials = 200
maxSweeps = 2000

order = 8
degree = 2
maxGroupSize = order//2  #NOTE: This is the block size needed to represent an arbitrary polynomial.

*_, Pi = riccati_matrices(order)
f = lambda xs: np.einsum('ni,ij,nj -> n', xs, Pi, xs)
sampleSizes = np.unique(np.geomspace(1e1, 1e6, numSteps).astype(int))
testSampleSize = int(1e6)

CACHE_DIRECTORY = CACHE_DIRECTORY.format(order=order, maxGroupSize=maxGroupSize)

test_points = 2*np.random.rand(testSampleSize,order)-1
test_measures = monomial_measures(test_points, degree)
test_values = f(test_points)


def sparse_dofs():
    return comb(degree+order-1, order-1)  # all polynomials of degree exactly `degree`

def bstt_dofs():
    bstt = random_homogenous_polynomial_v2([degree]*order, degree, maxGroupSize)
    return bstt.dofs()

ranks = random_homogenous_polynomial_v2([degree]*order, degree, maxGroupSize).ranks
def tt_dofs():
    bstt = random_full([degree]*order, ranks)
    return bstt.dofs()

def dense_dofs():
    return (degree+1)**order


console = Console()

console.rule(f"[bold]Riccati")

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


def multiIndices(_degree, _order):
    return filter(lambda mI: sum(mI) == _degree, product(range(_degree+1), repeat=_order))  # all polynomials of degree exactly `degree`


assert len(list(multiIndices(degree, order))) == sparse_dofs()


def residual(_bstt):
    return np.linalg.norm(_bstt.evaluate(test_measures) -  test_values) / np.linalg.norm(test_values)


def sparse_error(N):
    points = 2*np.random.rand(N,order)-1
    measures = monomial_measures(points, degree)
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


def bstt_error(N):
    bstt = random_homogenous_polynomial_v2([degree]*order, degree, maxGroupSize)
    points = 2*np.random.rand(N,order)-1
    measures = monomial_measures(points, degree)
    values = f(points)
    solver = ALS(bstt, measures, values)
    solver.maxSweeps = maxSweeps
    solver.targetResidual = 1e-16
    solver.run()
    return residual(solver.bstt)


def tt_error(N):
    bstt = random_full([degree]*order, ranks)
    points = 2*np.random.rand(N,order)-1
    measures = monomial_measures(points, degree)
    values = f(points)
    solver = ALS(bstt, measures, values)
    solver.maxSweeps = maxSweeps
    solver.targetResidual = 1e-16
    solver.run()
    return residual(solver.bstt)


os.makedirs(CACHE_DIRECTORY, exist_ok=True)
def compute(_error, _sampleSizes, _numTrials):
    cacheFile = f'{CACHE_DIRECTORY}/{_error.__name__}.npz'
    try:
        z = np.load(cacheFile)
        if 'errors_exact' in z.keys():
            print(f"WARNING: Old format. Converting...")
            np.savez_compressed(cacheFile, errors=z['errors_exact'], sampleSizes=z['sampleSizes'])
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
