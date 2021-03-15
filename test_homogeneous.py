from math import comb

import numpy as np
import xerus as xe
from numpy.polynomial.legendre import legval
from rich.console import Console
from rich.table import Table

from misc import random_homogenous_polynomial, random_homogenous_polynomial_v2, max_group_size
from als import ALS
from riccati import riccati_matrices


# ==========
#  TRAINING
# ==========

N = int(1e3)  # number of samples
# N = int(1e5)  # number of samples


# problem = "synthetic function"
# f = lambda xs: np.linalg.norm(xs, axis=1)**2  # (not homogeneous)
# f = lambda xs: xs[:,0]**2                     # (not homogeneous)
# f = lambda xs: legval(xs[:,0], [0,0,1])       # L2(x0)*L0(x1)*L0(x2) (homogeneous)
# f = lambda xs: sum(legval(xs[:,m], [0,0,1]) for m in range(xs.shape[1]))  # L2(x0)*L0(x1)*L0(x2) (homogeneous)
# M = 6
# univariateDegree = 6
# maxDegree = 2
# maxGroupSize = 1

# def measures(_points, _degree):
#     factors = np.sqrt(2*np.arange(_degree+1)+1)
#     return legval(_points, np.diag(factors)).T


problem = "value function"
# M = 32    # order
M = 8     # order
*_, Pi = riccati_matrices(M)
f = lambda xs: np.einsum('ni,ij,nj -> n', xs, Pi, xs)
univariateDegree = 3
maxDegree = 2
# maxGroupSize = 1
# maxGroupSize = 2
# maxGroupSize = 3
maxGroupSize = 4
# maxGroupSize = np.inf

def measures(_points, _degree):
    return _points.T[...,None]**np.arange(_degree+1)[None,None]


print(f"Recovery: {problem}")
print(f"    Order:                       {M}")
print(f"    Univariate degree:           {univariateDegree}")
print(f"    Homogeneous degree:          {maxDegree}")
print(f"    Maximal group size:          {maxGroupSize}")
print(f"    Maximal possible group size: {max_group_size(M, maxDegree)}")
print(f"    Number of samples:           {N}")


bstt = random_homogenous_polynomial_v2([univariateDegree]*M, maxDegree, maxGroupSize)
d = univariateDegree+1

samples = 2*np.random.rand(N,M)-1
meas = measures(samples, univariateDegree)
assert meas.shape == (M,N,d)
values = f(samples)
assert values.shape == (N,)

solver = ALS(bstt, meas, values, _verbosity=2)
solver.targetResidual = 1e-12
solver.run()


# =========
#  TESTING
# =========

def test(_bstt):
    N = int(1e3)  # number of test samples
    samples = 2*np.random.rand(N,M)-1
    meas = measures(samples, univariateDegree)
    assert meas.shape == (M,N,d)
    values = f(samples)
    assert values.shape == (N,)
    return ALS(_bstt, meas, values).residual()

def to_xe(_tt):
    ret = xe.TTTensor(_tt.dimensions)
    for pos in range(_tt.order):
        ret.set_component(pos, xe.Tensor.from_ndarray(_tt.components[pos]))
    return ret

def from_xe(_tt):
    from bstt import BlockSparseTT
    from misc import block
    order = _tt.order()
    dimensions = _tt.dimensions
    ranks = [1] + _tt.ranks() + [1]
    blocks = [[block[0:ranks[m],0:dimensions[m],0:ranks[m+1]]] for m in range(order)]
    components = [_tt.get_component(m).to_ndarray() for m in range(order)]
    return BlockSparseTT(components, blocks)

def dofs(_tt):
    return sum(_tt.get_component(pos).size for pos in range(_tt.order()))

def sparse_dofs(_tt):
    assert _tt.dimensions == [_tt.dimensions[0]]*_tt.order()
    degree = _tt.dimensions[0]-1
    return comb(degree+_tt.order()-1, _tt.order()-1)  # compare to test_homogeneous_ml.py

def dense_dofs(_tt):
    return np.product(_tt.dimensions)

xett = to_xe(solver.bstt)

console = Console()
table = Table(title=f"{problem.lower().capitalize()} (maxGroupSize: {maxGroupSize})", title_style="bold", show_header=True, header_style="dim")
table.add_column("", style="dim", justify="left")
table.add_column("Error", justify="right")
table.add_column("Dofs", justify="right")
table.add_row("BSTT", f"{test(solver.bstt):.2e}", f"{solver.bstt.dofs()}")
xett.round(1e-12)
table.add_row("rounded TT (1e-12)", f"{test(from_xe(xett)):.2e}", f"{dofs(xett)}")
xett.round(1e-8)
table.add_row("rounded TT (1e-8)", f"{test(from_xe(xett)):.2e}", f"{dofs(xett)}")
xett.round(1e-6)
table.add_row("rounded TT (1e-6)", f"{test(from_xe(xett)):.2e}", f"{dofs(xett)}")
table.add_row("sparse", f"", f"{sparse_dofs(xett)}")
table.add_row("full", f"", f"{dense_dofs(xett)}")
console.print(table)
