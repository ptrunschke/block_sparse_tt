from math import comb

import numpy as np
import xerus as xe
from rich.console import Console
from rich.table import Table

from misc import random_homogenous_polynomial
from als import ALS
from riccati import riccati_matrices


# ==========
#  TRAINING
# ==========

N = int(1e3)  # number of samples

M = 32    # order
*_, Pi = riccati_matrices(M)
f = lambda xs: np.einsum('ni,ij,nj -> n', xs, Pi, xs)
univariateDegree = 2
homogeneousDegree = 2

bstt = random_homogenous_polynomial([univariateDegree]*M, homogeneousDegree, 1)
print("Dimensions:", bstt.dimensions)
print("Ranks:     ", bstt.ranks)
d = univariateDegree+1

samples = 2*np.random.rand(N,M)-1
measures = samples.T[...,None]**np.arange(d)[None,None]
assert measures.shape == (M,N,d)
values = f(samples)
assert values.shape == (N,)

solver = ALS(bstt, measures, values, _verbosity=1)
solver.targetResidual = 1e-12
solver.run()


# =========
#  TESTING
# =========

def test(_bstt):
    N = int(1e3)  # number of test samples
    samples = 2*np.random.rand(N,M)-1
    factors = np.sqrt(2*np.arange(d)+1)
    measures = samples.T[...,None]**np.arange(d)[None,None]
    assert measures.shape == (M,N,d)
    values = f(samples)
    assert values.shape == (N,)
    return ALS(_bstt, measures, values).residual()

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
    return comb(degree+_tt.order(), _tt.order())

def dense_dofs(_tt):
    return np.product(_tt.dimensions)

xett = to_xe(solver.bstt)

console = Console()
table = Table(title="Quardartic polynomial", title_style="bold", show_header=True, header_style="dim")
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
