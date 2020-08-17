#NOTE: This implementation is not meant to be memory efficient or fast but rather to test the approximation capabilities of the proposed model class.
import numpy as np
from numpy.polynomial.legendre import legval
from scipy.sparse import block_diag

class BlockSparseTT(object):
    def __init__(self, _components, _blocks):
        """
        _components : list of ndarrays of order 3
            The list of component tensors for the TTTensor.
        _blocks : list of list of triples
            For the k-th component tensor _blocks[k] contains the list of its blocks of non-zero values.
            Each block is represented by a triple of integers and slices.
            To obtain the block This triple the slice in the component tensor.

        NOTE: Later we can remove _components and augment each triple in _blocks by an array that contains the data in this block.
        """
        assert all(cmp.ndim == 3 for cmp in _components)
        assert _components[0].shape[0] == 1
        assert all(cmp1.shape[2] == cmp2.shape[0] for cmp1,cmp2 in zip(_components[:-1], _components[1:]))
        assert _components[-1].shape[2] == 1
        self.components = _components
        assert isinstance(_blocks, list) and len(_blocks) == self.order

        assert all(isinstance(compBlocks, list) for compBlocks in _blocks)
        assert all(isinstance(block, tuple) and len(block) == 3 for compBlocks in _blocks for block in compBlocks)
        for compBlocks in _blocks:
            for block in compBlocks:
                for slc in block:
                    assert isinstance(slc, slice) and isinstance(slc.start, int) and isinstance(slc.stop, int) and 0 <= slc.start < slc.stop and slc.step is None, slc
        #NOTE: The final two conditions may restrict the structure of the blocks unnecessarily.

        def disjoint_slices(_slc1, _slc2):
            return (_slc1.start <= _slc2.start and _slc1.stop <= _slc2.start) or _slc2.stop <= _slc1.start

        def disjoint_blocks(_block1, _block2):
            return any(disjoint_slices(slc1, slc2) for slc1,slc2 in zip(_block1, _block2))

        for m, (comp, compBlocks) in enumerate(zip(self.components, _blocks)):
            for i in range(len(compBlocks)):
                for j in range(i):
                    assert disjoint_blocks(compBlocks[i], compBlocks[j])
                    assert compBlocks[i][0] == compBlocks[j][0] or disjoint_slices(compBlocks[i][0], compBlocks[j][0]), f"Component {m}: {compBlocks[i][0]} vs {compBlocks[j][0]}"
                    assert compBlocks[i][2] == compBlocks[j][2] or disjoint_slices(compBlocks[i][2], compBlocks[j][2]), f"Component {m}: {compBlocks[i][2]} vs {compBlocks[j][2]}"
                    #NOTE: Requiring the left and right slices to be identical or disjoint (i.e. dont overlap partially) is not a restriction.
                    #      If this was the case then one could split every two overlapping slices into 3 new slices.
                    #      The first contains the left non-overlapping part, the second contains the overlapping part and the third contains the right non-overlapping part.
            #NOTE: This can be done more efficiently.
            l = np.full((comp.shape[0],), False, dtype=bool)
            e = np.full((comp.shape[1],), False, dtype=bool)
            r = np.full((comp.shape[2],), False, dtype=bool)
            for block in compBlocks:
                assert (m == 0 or block[0].stop <= self.ranks[m-1]) and block[1].stop <= self.dimensions[m] and (m == self.order-1 or block[2].stop <= self.ranks[m]), f"NOT {block} < {(self.ranks[m-1], self.dimensions[m], self.ranks[m])}"
                #NOTE: Ensure that the slices are not to large for the given ranks. (Recall that slice.stop is an exclusive bound,)
                l[block[0]] = True
                e[block[1]] = True
                r[block[2]] = True
            assert np.all(l) and np.all(e) and np.all(r), f"Blocks for component {m} leave a hole ({l}, {e}, {r})."
            #NOTE: Check for "holes" left by the blocks. These holes mean that there can be no SVD that preserves the sparsity structure.
            #NOTE: The condition on block[1] may restrict the structure of the blocks unnecessarily.

        leftSlices = []
        rightSlices = []
        for m in range(self.order):
            leftSlices_m = set()
            rightSlices_m = set()
            for block in _blocks[m]:
                leftSlices_m.add((block[0].start, block[0].stop, block[0].step))
                rightSlices_m.add((block[2].start, block[2].stop, block[2].step))
            leftSlices.append(leftSlices_m)
            rightSlices.append(rightSlices_m)

        assert leftSlices[0] == {(0,1,None)}
        for m in range(self.order-1):
            assert rightSlices[m] == leftSlices[m+1]
        assert rightSlices[-1] == {(0,1,None)}
        #NOTE: Check that for every two neighboring components every block in the left component has a corresponding block in the right component.
        #      Otherwise, if for example the right component would be dense, the left component must be dense as well after a core move to the left.

        def size(_slc):
            return _slc.stop - _slc.start

        for d in range(self.order):
            if d > 0:
                for slcTpl in leftSlices[d]:
                    m = slcTpl[1] - slcTpl[0]
                    n = 0
                    for blk in _blocks[d]:
                        if blk[0].start == slcTpl[0]:
                            n += size(blk[1]) * size(blk[2])
                    assert m <= n, f"NOT {m} <= {n}"
                    assert m <= n, f"Too few non-zero columns in 0-unfolded submatrix components[{d}][{slcTpl[0]}:{slcTpl[1]}]: NOT m <= n (rows: m={m}, columns: n={n})"
            if d < self.order-1:
                for slcTpl in rightSlices[d]:
                    m = 0
                    for blk in _blocks[d]:
                        if blk[2].start == slcTpl[0]:
                            m += size(blk[0]) * size(blk[1])
                    n = slcTpl[1] - slcTpl[0]
                    assert m >= n, f"Too few non-zero rows in (0,1)-unfolded submatrix components[{d}][:,{slcTpl[0]}:{slcTpl[1]}]: NOT m >= n (rows: m={m}, columns: n={n})"
        #TODO: Now check that the slices are not too large.
        #      Imagine a block of the form ((0:10), 0, 0). Obviously, there cannot be an SVD that preserves the sparsity structure.

        self.blocks = _blocks

        leftBlocks = []
        for d in range(self.order):
            #NOTE: When the left blocks are used the core (=components[d]) is always reshaped into shape (core.shape[0], -1).
            #      leftBlocks[d][k] is used to select the columns of this matrification for the k-th slice in the rows.
            leftBlock_d = []
            indices = np.arange(np.product(self.components[d].shape[1:])).reshape(self.components[d].shape[1:])
            for slcTpl in sorted(leftSlices[d], key=lambda _slcTpl: _slcTpl[0]):
                slcIndices = [indices[blk[1], blk[2]].reshape(-1) for blk in _blocks[d] if blk[0].start == slcTpl[0]]
                leftBlock_d.append((slice(*slcTpl), np.sort(np.concatenate(slcIndices))))
            leftBlocks.append(leftBlock_d)
        self.leftBlocks = leftBlocks

        rightBlocks = []
        for d in range(self.order):
            #NOTE: When the right blocks are used the core (=components[d]) is always reshaped into shape (-1, core.shape[2]).
            #      rightBlocks[d][k] is used to select the rows of this matrification for the k-th slice in the columns.
            rightBlock_d = []
            indices = np.arange(np.product(self.components[d].shape[:2])).reshape(self.components[d].shape[:2])
            for slcTpl in sorted(rightSlices[d], key=lambda _slcTpl: _slcTpl[0]):
                slcIndices = [indices[blk[0], blk[1]].reshape(-1) for blk in _blocks[d] if blk[2].start == slcTpl[0]]
                rightBlock_d.append((np.sort(np.concatenate(slcIndices)), slice(*slcTpl)))
            rightBlocks.append(rightBlock_d)
        self.rightBlocks = rightBlocks

        self.__corePosition = None
        self.verify()

    def verify(self):
        for e, (compBlocks, component) in enumerate(zip(self.blocks, self.components)):
            assert np.all(np.isfinite(component))
            cmp = np.array(component)
            for block in compBlocks:
                cmp[block] = 0
            assert np.allclose(cmp, 0), f"Component {e} does not satisfy the block structure. Error:\n{np.max(abs(cmp))}"

    @property
    def corePosition(self):
        return self.__corePosition

    def assume_corePosition(self, _position):
        assert 0 <= _position and _position < self.order
        self.__corePosition = _position

    @property
    def ranks(self):
        return [cmp.shape[2] for cmp in self.components[:-1]]

    @property
    def dimensions(self):
        return [cmp.shape[1] for cmp in self.components]

    @property
    def order(self):
        return len(self.components)

    def move_core(self, _direction):
        assert isinstance(self.corePosition, int)
        assert _direction in ['left', 'right']
        # SVD for the LEFT move
        # =====================
        # Consider the component tensor M of shape (r1,d,r2) and consider that this tensor contains non-zero blocks
        # at the 3D-slices ((:a), scl_11[k], slc_12[k]) for k=1,...,K and ((a:), slc_21[l], scl_22[l]) for l=1,...,L.
        # After a 1-matricisation we obtain a matrix 
        #    ┌       ┐
        #    │ M[:a] │
        #    │ M[a:] │
        #    └       ┘
        # of shape (r1, d*r2) and the slices take the form ((:a), scl_1[k]) and ((a:), slc_2[l]) where slc_1 and scl_2
        # just select certain columns of this matricisation.
        # Our goal is to obtain an SVD-like decomposition of this matrix M = U S Vt such that the sparsity pattern is preserved.
        # By this we mean that U is a block-diagonal matrix, so that it can be multiplied to the component tensor to the
        # left without modifying its sparsity structure, that S is a diagonal matrix (as usual) and that Vt only has 
        # non-zero entries in the blocks ((:a), scl_1[k]) and ((a:), slc_2[l]).
        # Let M[:a] = UₐSₐVtₐ and M[a:] = UᵃSᵃVtᵃ. Then such a decomposition is given by
        #    ┌       ┐ ┌       ┐ ┌     ┐
        #    │ Uₐ    │ │ Sₐ    │ │ Vtₐ │
        #    │    Uᵃ │ │    Sᵃ │ │ Vtᵃ │
        #    └       ┘ └       ┘ └     ┘
        # To see that this preserves tha sparsity pattern in Vtₐ and Vtᵃ we focus on Vtₐ.
        # Define a permutation matrix Pₐ that sorts the the columns of M[:a] such that M[:a] Pₐ = [ 0 X ], perform an SVD
        # on X = Uˣ Sˣ Vtˣ and observe that M[:a] = Uˣ Sˣ [ 0 Vtˣ ] Ptₐ.
        # NOTE: This not only proves the claim but also shows a more performant way to compute the SVD.

        #TODO: Das gilt allerdings nur für exakte arithmetik.
        # Sei M ein Vektor der Dimension 2 und betrachte folgendes Beispiel in Computerarithmetik
        #                   ┌         ┐ 
        #    ┌       ┐      │ 0   1   │ ┌     ┐ 
        #    │       │  ==  │ U₁₁ U₁₂ │ │ 1   │ 
        #    │ M M+ε │      │ U₂₁ U₂₂ │ │   ε │ Vt
        #    └       ┘      └         ┘ └     ┘ 
        # Diese Beispiel zeigt, dass die oben vorgestellte methode nicht nur effizienter ist, sonder auch numerisch stabiler.
        if _direction == 'left':
            assert 0 < self.corePosition
            oldCore = self.components[self.corePosition]
            newCore = self.components[self.corePosition-1]
            test = np.einsum('ler,rds', newCore, oldCore)

            oldCore_shape = oldCore.shape
            oldCore = oldCore.reshape(oldCore_shape[0], -1)

            US_blocks, Vt_blocks = [], []
            for leftBlock in self.leftBlocks[self.corePosition]:
                X = oldCore[leftBlock]
                u,s,vt = np.linalg.svd(X, full_matrices=False)
                assert u.shape[0] == u.shape[1]  #TODO: Handle the case that a singular value is zero.
                US_blocks.append(u*s)
                Vt_blocks.append(vt)

            US = block_diag(US_blocks, format='csc')
            Vt = np.zeros(oldCore.shape)
            for leftBlock, Vt_block in zip(self.leftBlocks[self.corePosition], Vt_blocks):
                Vt[leftBlock] = Vt_block

            oldCore = Vt.reshape(oldCore_shape)
            newCore = (newCore.reshape(-1, newCore.shape[2]) @ US).reshape(newCore.shape)

            oldCore_test = np.array(oldCore)
            for blk in self.blocks[self.corePosition]:
                oldCore_test[blk] = 0
            assert np.allclose(oldCore_test, 0)
            assert np.allclose(test, np.einsum('ler,rds', newCore, oldCore))

            self.components[self.corePosition-1] = newCore
            self.components[self.corePosition] = oldCore
            self.__corePosition -= 1
        else:
            assert self.corePosition < self.order-1
            oldCore = self.components[self.corePosition]
            newCore = self.components[self.corePosition+1]
            test = np.einsum('ler,rds', oldCore, newCore)

            oldCore_shape = oldCore.shape
            oldCore = oldCore.reshape(-1, oldCore_shape[2])

            U_blocks, SVt_blocks = [], []
            for rightBlock in self.rightBlocks[self.corePosition]:
                X = oldCore[rightBlock]
                u,s,vt = np.linalg.svd(X, full_matrices=False)
                assert vt.shape[0] == vt.shape[1]  #TODO: Handle the case that a singular value is zero.
                U_blocks.append(u)
                SVt_blocks.append(s[:,None]*vt)

            U = np.zeros(oldCore.shape)
            for rightBlock, U_block in zip(self.rightBlocks[self.corePosition], U_blocks):
                U[rightBlock] = U_block
            SVt = block_diag(SVt_blocks, format='csr')

            oldCore = U.reshape(oldCore_shape)
            newCore = (SVt @ newCore.reshape(newCore.shape[0], -1)).reshape(newCore.shape)

            oldCore_test = np.array(oldCore)
            for blk in self.blocks[self.corePosition]:
                oldCore_test[blk] = 0
            assert np.allclose(oldCore_test, 0)
            assert np.allclose(test, np.einsum('ler,rds', oldCore, newCore))

            self.components[self.corePosition] = oldCore
            self.components[self.corePosition+1] = newCore
            self.__corePosition += 1
        self.verify()

    @classmethod
    def random(cls, _dimensions, _ranks, _blocks):
        ranks = [1] + _ranks + [1]
        components = [np.zeros((leftRank, dimension, rightRank)) for leftRank, dimension, rightRank in zip(ranks[:-1], _dimensions, ranks[1:])]
        for comp, compBlocks in zip(components, _blocks):
            for block in compBlocks:
                comp[block] = np.random.randn(*comp[block].shape)
        return cls(components, _blocks)

class ALS(object):
    def __init__(self, _bstt, _measurements, _values):
        self.bstt = _bstt
        assert len(_measurements) == self.bstt.order
        assert all(np.shape(compMeas) == (len(_values), dim) for compMeas, dim in zip(_measurements, self.bstt.dimensions))
        self.measurements = _measurements
        self.values = _values
        self.maxSweeps = 100
        self.targetResidual = 1e-4
        self.minDecrease = 1e-2

        self.leftStack = [np.ones((len(self.values),1))] + [None]*(self.bstt.order-1)
        self.rightStack = [np.ones((len(self.values),1))]
        self.bstt.assume_corePosition(bstt.order-1)
        while self.bstt.corePosition > 0:
            self.move_core('left')

    def move_core(self, _direction):
        self.bstt.move_core(_direction)
        if _direction == 'left':
            print(f"move_core: {self.bstt.corePosition+1} --> {self.bstt.corePosition}")
            self.leftStack.pop()
            self.rightStack.append(np.einsum('ler, ne, nr -> nl', self.bstt.components[self.bstt.corePosition+1], self.measurements[self.bstt.corePosition+1], self.rightStack[-1]))
        elif _direction == 'right':
            print(f"move_core: {self.bstt.corePosition-1} --> {self.bstt.corePosition}")
            self.rightStack.pop()
            self.leftStack.append(np.einsum('nl, ne, ler -> nr', self.leftStack[-1], self.measurements[self.bstt.corePosition-1], self.bstt.components[self.bstt.corePosition-1]))
        else:
            raise ValueError(f"Unknown _direction. Expected 'left' or 'right' but got '{_direction}'")

    def residual(self):
        pos = self.bstt.corePosition
        core = self.bstt.components[pos]
        L = self.leftStack[-1]
        E = self.measurements[pos]
        R = self.rightStack[-1]
        pred = np.einsum('ler,nl,ne,nr -> n', core, L, E, R)
        return np.linalg.norm(pred -  self.values) / np.linalg.norm(self.values)

    def microstep(self):
        pos = self.bstt.corePosition
        core = self.bstt.components[pos]
        coreBlocks = self.bstt.blocks[pos]
        L = self.leftStack[-1]
        E = self.measurements[pos]
        R = self.rightStack[-1]

        Op_blocks = []
        Param_blocks = []
        for block in coreBlocks:
            op = np.einsum('nl,ne,nr -> nler', L[:, block[0]], E[:, block[1]], R[:, block[2]])
            op.shape = len(self.values), -1
            Op_blocks.append(op)
            Param_blocks.append(core[block].reshape(-1))
        Op = np.concatenate(Op_blocks, axis=1)
        Param = np.concatenate(Param_blocks)
        Res = np.linalg.solve(Op.T @ Op, Op.T @ self.values)

        def slice_size(_slc):
            return _slc.stop - _slc.start

        def block_size(_blk):
            assert len(_blk) == 3
            return slice_size(_blk[0])*slice_size(_blk[1])*slice_size(_blk[2])

        a = 0
        for block in coreBlocks:
            o = a + block_size(block)
            core[block].reshape(-1)[:] = Res[a:o]
            a = o

    def run(self):
        prev_residual = self.residual()
        print(f"Initial residuum: {prev_residual:.2e}")
        for sweep in range(self.maxSweeps):
            while self.bstt.corePosition < self.bstt.order-1:
                self.microstep()
                self.move_core('right')
            while self.bstt.corePosition > 0:
                self.microstep()
                self.move_core('left')

            residual = self.residual()
            print(f"[{sweep}] Residuum: {residual:.2e}")

            if residual < self.targetResidual:
                print(f"Terminating (targetResidual reached)")
                print(f"Final residuum: {self.residual():.2e}")
                return

            assert residual <= prev_residual
            if (prev_residual - residual) < self.minDecrease*residual:
                print(f"Terminating (minDecrease reached)")
                print(f"Final residuum: {self.residual():.2e}")
                return

            prev_residual = residual

        print(f"Terminating (maxSweeps reached)")
        print(f"Final residuum: {self.residual():.2e}")


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


# def random_polynomial(_univariateDegrees, _maxTotalDegree):
#     assert _maxTotalDegree >= max(_univariateDegrees)
#     blockSlices = []
#     d = 0
#     blockSlices.append([slice(k,k+1) for k in range(_univariateDegrees[d])])
#     d = 1
#     for k in range(len(blockSlices[-1])):
#         for l in range(_univariateDegrees[d]):
#             # left block: slice_size(blockSlices[-1][k]) of degree k
#             # central block: degree l
#             now we have to think about how many polynomials we allow. This has to do with the rank.
#             for every polynomial of degree k

bstt = BlockSparseTT.random([5,5,5,5], [5,8,5], [
        [block[0,l,l] for l in range(5)],
        [block[0,0,0]] +  # given only one polynomial P(x) of degree 0 you can also only create one polynomial Q(x,y) = P(x) of degree zero.
            [block[k,l,1+2*(k+l-1):1+2*(k+l)] for k in range(5) for l in range(5) if 0 < k+l < 4] +
            [block[k,l,1+2*(4-1)] for k in range(5) for l in range(5) if k+l == 4],
        [block[0,l,l] for l in range(5)] +
            [block[1+2*(k-1):1+2*k, l, k+l] for k in range(1,4) for l in range(5) if k+l < 5] +
            [block[1+2*(4-1), l, 4+l] for l in range(5) if 4+l < 5],
        [block[k,l,0] for k in range(5) for l in range(5) if k+l < 5]
    ])

# bstt.assume_corePosition(bstt.order-1)
# while bstt.corePosition > 0:
#     print(f"move_core: {bstt.corePosition} --> {bstt.corePosition-1}")
#     bstt.move_core('left')
# while bstt.corePosition < bstt.order-1:
#     print(f"move_core: {bstt.corePosition} --> {bstt.corePosition+1}")
#     bstt.move_core('right')
# exit()


N = int(1e3)  # number of samples
f = lambda xs: 1/(1+25*np.linalg.norm(xs, axis=1)**2)

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
solver.run()







# Consider the SVD of the matrix
#    ┌    ╶┐
#    │ A C │
#    │ B D │
#    └     ┘
# Then a SVD-like decomposition of this matrix is given by
#                    ┌            ╶┐ ┌        ╶┐
#                    │ Sᴬ          │ │ Vtᴬ     │
#    ┌            ╶┐ │    Sᴮ       │ │ Vtᴮ     │
#    │ Uᴬ    Uᶜ    │ │       Sᶜ    │ │     Vtᶜ │
#    │    Uᴮ    Uᴰ │ │          Sᴰ │ │     Vtᴰ │
#    └             ┘ └             ┘ └         ┘
# with A = UᴬSᴬVtᴬ, B = UᴮSᴮVtᴮ, ... the SVDs of A, B, ...
# It is easy to see that the middle factor matrix is a diagonal matrix
# and that the left and right factor matrices are orthogonal.
# As an example consider
#    ┌        ╶┐                     ┌                        ╶┐
#    │ Vtᴬ     │                     │ VtᴬVᴬ                   │
#    │ Vtᴮ     │ ┌            ╶┐     │       VtᴮVᴮ             │
#    │     Vtᶜ │ │ Vᴬ Vᴮ       │  =  │             VtᶜVᶜ       │  =  I
#    │     Vtᴰ │ │       Vᶜ Vᴰ │     │                   VtᴰVᴰ │
#    └         ┘ └             ┘     └                         ┘
