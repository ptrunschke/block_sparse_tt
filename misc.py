import numpy as np
from numpy.polynomial.legendre import legval

from bstt import Block, BlockSparseTT

import autoPDB


class __block(object):
    def __getitem__(self, slices):
        assert len(slices) == 3
        assert all(isinstance(slc, (int, slice)) for slc in slices)
        def as_slice(_slc):
            if isinstance(_slc, slice):
                return _slc
            assert 0 <= _slc
            return slice(_slc, _slc+1)
        return Block(as_slice(slc) for slc in slices)
block = __block()


def random_polynomial(_univariateDegrees, _maxTotalDegree):
    assert _maxTotalDegree >= max(_univariateDegrees)  #TODO: This is only required due to unnecessary restrictions in BlockSparseTT.
    # Consider a coefficient tensor with dimensions [3,3,3,3] for a polynomial with (total) degree 2.
    # (Note that the space of univariate polynomials of degree 2 has dimension 3.)
    # Degree 2 is the lowest degree that allows for combination of multiple modes.
    # (A polynomial of degree 2 can be created either by P1(x)*P1(y) or by P0(x)*P2(y). For degree 0 you only have P0(x)*P0(y) and for degree 1 you only have P0(x)*P1(y).)
    # Order 4 is the smallest order that allows for a TT-rank larger than 3.
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


# def random_nearest_neighbor_polynomial(_univariateDegrees, _nnranks):
#     dimensions = [dim+1 for dim in _univariateDegrees]
#     nnslice = np.concatenate([0], np.cumsum(np.concatenate([1], _nnranks)))
#     nnslice = [slice(s1, s2) for s1,s2 in zip(nnslice[:-1], nnslice[1:])]
#     dimslice = [slice(0, dim) for dim in dimensions]

#     ranks = []
#     blocks = []
#     currentMaxNeighbors = 0
#     maxNeighbors = len(_nnranks)
#     for m in range(len(_dimensions)-1):
#         blocks_m = []
#         blocks_m.append(block[nnslice[0], 0, nnslice[0]])
#         for n in range(min(currenMaxNeighbors, maxNeighbors-1)):
#             blocks_m.append(block[nnslice[n], dimslice[m], nnslice[n+1]])
#         if currenMaxNeighbors == maxNeighbors:
#             blocks_m.append(block[nnslice[-1], 0, nnslice[-1]])
#         ranks.append(max(blk[2].stop for blk in blocks_m))
#         blocks.append(blocks_m)
#         currenMaxNeighbors = min(currenMaxNeighbors+1, len(_nnranks))
#     blocks.append([block[nnslice[-1],0,0], block[nnslice[-2],:,0]])
#     return BlockSparseTT.random(dimensions, ranks, blocks)


def random_full(_univariateDegrees, _rank):
    """
    Create a randomly initialized TT with the given rank.

    Note that this TT  will not have sparse blocks!
    """
    order = len(_univariateDegrees)
    dimensions = [dim+1 for dim in _univariateDegrees]
    maxTheoreticalRanks = np.minimum(np.cumprod(dimensions[:-1]), np.cumprod(dimensions[1:][::-1])[::-1])
    ranks = [1] + np.minimum(maxTheoreticalRanks, _rank).tolist() + [1]
    blocks = [[block[0:ranks[m],0:dimensions[m],0:ranks[m+1]]] for m in range(order)]
    return BlockSparseTT.random(dimensions, ranks[1:-1], blocks)
