import numpy as np
from numpy.polynomial.legendre import legval

from bstt import BlockSparseTT
from als import ALS


class __block(object):
    def __getitem__(self, slices):
        assert len(slices) == 3
        assert all(isinstance(slc, (int, slice)) for slc in slices)
        def as_slice(_slc):
            if isinstance(_slc, slice):
                return _slc
            assert 0 <= _slc
            return slice(_slc, _slc+1)
        return tuple(as_slice(slc) for slc in slices)
block = __block()


def random_polynomial(_univariateDegrees, _maxTotalDegree):
    assert _maxTotalDegree >= max(_univariateDegrees)  #TODO: This is only required due to unnecessary restrictions in BlockSparseTT.

    # Consider a tensor with dimensions [3,3,3,3] and a total degree of 2.
    # The chosen univariate degree 2 (<-> dimension 3) is the lowest degree that allows for the combination
    # of more that two different univariate polynomials to obtain a multivariate polynomial of degree 2.
    # (A polynomial of degree 2  can be created either by P1(x)*P1(y) or by P0(x)*P2(y). For degree 0 and 1 you have less options.)
    # Order 4 is the smallest order that allows for a TT-rank larger than 3.
    # The first component tensor has the non-zero blocks:
    #     (0,0,0)
    #     (0,1,1)
    #     (0,2,2)
    # The second component has the blocks:
    #     (0,0,0)
    #     (0,1,1)
    #     (0,2,2)
    #     (1,0,1)
    #     (1,1,2)
    #     (2,0,2)
    # And so on...
    # The polynomials that are allowed by this scheme are:
    #     P0(x)*P0(y)*P0(z)*P0(w) with coefficient C0000 = C0[0,0,0]*C1[0,0,0]*C2[0,0,0]*C3[0,0,0]
    #     P1(x)*P0(y)*P0(z)*P0(w) with coefficient C1000 = C0[0,1,1]*C1[1,0,1]*C2[1,0,1]*C3[1,0,0]
    #     P0(x)*P1(y)*P0(z)*P0(w) with coefficient C0100 = C0[0,0,0]*C1[0,1,1]*C2[1,0,1]*C3[1,0,0]
    #     ...
    #     P2(x)*P0(y)*P0(z)*P0(w) with coefficient C2000 = C0[0,2,2]*C1[2,0,2]*C2[2,0,2]*C3[2,0,0]
    #     ...
    #     P1(x)*P1(y)*P0(z)*P0(w) with coefficient C2000 = C0[0,1,1]*C1[1,1,2]*C2[2,0,2]*C3[2,0,0]
    #     ...
    # From the table above it is easy to see that each polynomial has its own coefficient.
    # Starting with a polynomial of degree zero the components C{k}[0,0,0] are chosen arbitrarily to satisfy the above equation.
    # Then the C{k}[0,1,1] and C{k}[1,0,1] are chosen for polynomials of degree 1. For polynomials of degree 2 again new non-zero 
    # components can be chosen and so on.
    # Each coefficient C{i}{j}{k}{l} can be obtained since in every product at least one new coefficient appears.
    # All blocks that are zero are only necessary to represent polynomials of higher degree.
    #NOTE: This shows that in this specific setting a rank of 3 is sufficient to obtain all polynomials of degree 2
    #      and it hints to the fact that a polynomial of degree P can bre represented by a TT of rank P+1.

    dimensions = [deg+1 for deg in _univariateDegrees]
    ranks = []
    blocks = []
    blocks.append([block[0,l,l] for l in range(dimensions[0]) if l <= _maxTotalDegree])
    ranks.append(min(dimensions[0]-1, _maxTotalDegree)+1)
    for m in range(1, len(_univariateDegrees)-1):
        blocks.append([block[k,l,k+l] for k in range(ranks[-1]) for l in range(dimensions[m]) if k+l <= _maxTotalDegree])
        ranks.append(min(ranks[-1]-1 + dimensions[m]-1, _maxTotalDegree)+1)
    blocks.append([block[k,l,0] for k in range(ranks[-1]) for l in range(dimensions[-1]) if k+l <= _maxTotalDegree])
    return BlockSparseTT.random(dimensions, ranks, blocks)


def random_nearest_neighbor_polynomial(_univariateDegrees, _nnranks):
    dimensions = [dim+1 for dim in _univariateDegrees]
    nnslice = np.concatenate([0], np.cumsum(np.concatenate([1], _nnranks)))
    nnslice = [slice(s1, s2) for s1,s2 in zip(nnslice[:-1], nnslice[1:])]
    dimslice = [slice(0, dim) for dim in dimensions]

    ranks = []
    blocks = []
    currentMaxNeighbors = 0
    maxNeighbors = len(_nnranks)
    for m in range(len(_dimensions)-1):
        blocks_m = []
        blocks_m.append(block[nnslice[0], 0, nnslice[0]])
        for n in range(min(currenMaxNeighbors, maxNeighbors-1)):
            blocks_m.append(block[nnslice[n], dimslice[m], nnslice[n+1]])
        if currenMaxNeighbors == maxNeighbors:
            blocks_m.append(block[nnslice[-1], 0, nnslice[-1]])
        ranks.append(max(blk[2].stop for blk in blocks_m))
        blocks.append(blocks_m)
        currenMaxNeighbors = min(currenMaxNeighbors+1, len(_nnranks))
    blocks.append([block[nnslice[-1],0,0], block[nnslice[-2],:,0]])
    return BlockSparseTT.random(dimensions, ranks, blocks)



# ==========
#  TRAINING
# ==========

# bstt = random_polynomial([20,20,20,20], 40)
# bstt = random_polynomial([15,15,15,15], 15)
bstt = random_polynomial([7,7,7,7], 7)

N = int(1e5)  # number of samples
f = lambda xs: np.linalg.norm(xs, axis=1)**2
# f = lambda xs: 1/(1+25*np.linalg.norm(xs, axis=1)**2)

M = bstt.order
assert min(bstt.dimensions) == max(bstt.dimensions)
d = bstt.dimensions[0]

samples = 2*np.random.rand(N,M)-1
factors = np.sqrt(2*np.arange(d)+1)
measures = legval(samples, np.diag(factors)).T
assert measures.shape == (M,N,d)
values = f(samples)
assert values.shape == (N,)

solver = ALS(bstt, measures, values)
solver.minDecrease = 1e-4
solver.run()


# =========
#  TESTING
# =========

N = int(1e3)  # number of test samples
samples = 2*np.random.rand(N,M)-1
factors = np.sqrt(2*np.arange(d)+1)
measures = legval(samples, np.diag(factors)).T
assert measures.shape == (M,N,d)
values = f(samples)
assert values.shape == (N,)

tester = ALS(solver.bstt, measures, values)
print(f"Test set residual: {tester.residual():.2e}")
