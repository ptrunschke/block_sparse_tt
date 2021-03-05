import numpy as np
from scipy.sparse import block_diag, diags


class Block(tuple):
    @property
    def size(self):
        def slice_size(_slc):
            return _slc.stop - _slc.start
        ret = 1
        for slc in self:
            ret *= slice_size(slc)
        return ret

    def is_valid(self):
        for slc in self:
            if not (isinstance(slc, slice) and isinstance(slc.start, int) and isinstance(slc.stop, int) and 0 <= slc.start < slc.stop and slc.step in (None,1)):
                #NOTE: The final two conditions may restrict the structure of the blocks unnecessarily.
                return False
        return True

    def disjoint(self, _other):
        assert isinstance(_other, Block) and len(self) == len(_other) and self.is_valid() and _other.is_valid()
        def disjoint_slices(_slc1, _slc2):
            # return (_slc1.start <= _slc2.start and _slc1.stop <= _slc2.start) or _slc2.stop <= _slc1.start
            return _slc1.stop <= _slc2.start or _slc2.stop <= _slc1.start
        return any(disjoint_slices(slc1, slc2) for slc1,slc2 in zip(self, _other))

    def contains(self, _other):
        assert isinstance(_other, Block) and len(self) == len(_other) and self.is_valid() and _other.is_valid()
        def contains_slice(_slc1, _slc2):
            return _slc1.start <= _slc2.start and _slc1.stop >= _slc2.stop
        return any(contains_slice(slc1, slc2) for slc1,slc2 in zip(self, _other))

    def coherent(self, _other):
        """
        Tests if all blocks are defined by non-overlapping slices.
        This is not a restriction since every two overlapping slices can be split into three new slices:
        The first slice contains the left non-overlapping part, the second contains the overlapping part and the third contains the right non-overlapping part.
        """
        #NOTE: The condition that the middle slices are coherent may restrict TT-Blocks unnecessarily.
        assert isinstance(_other, Block) and len(self) == len(_other) and self.is_valid() and _other.is_valid()
        def disjoint_slices(_slc1, _slc2):
            return (_slc1.start <= _slc2.start and _slc1.stop <= _slc2.start) or _slc2.stop <= _slc1.start
        return all((slc1 == slc2 or disjoint_slices(slc1, slc2)) for slc1, slc2 in zip(self, _other))


class BlockSparseTensor(object):
    def __init__(self, _data, _blocks, _shape):
        assert isinstance(_data, np.ndarray) and _data.ndim == 1
        self.data = _data
        assert isinstance(_blocks, (list, tuple))
        self.blocks = [Block(block) for block in _blocks]
        assert all(block.is_valid() for block in self.blocks) and sum(block.size for block in self.blocks) == self.data.size
        for i in range(len(self.blocks)):
            for j in range(i):
                assert self.blocks[i].disjoint(self.blocks[j]) and self.blocks[i].coherent(self.blocks[j])
        assert isinstance(_shape, tuple) and np.all(np.array(_shape) > 0)
        shapeBlock = Block(slice(0,dim,1) for dim in _shape)
        assert all(shapeBlock.contains(block) for block in self.blocks)
        self.shape = _shape

    def dofs(self):
        return sum(blk.size for blk in self.blocks)

    def svd(self, _mode):
        """
        Perform an SVD along the `_mode`-th mode while retaining the the block structure.

        The considered matricisation has `_mode` as its rows and all other modes as its columns.
        If `U,S,Vt = X.svd(_mode)`, then `X == (U @ S) @[_mode] Vt` where `@[_mode]` is the contraction with the `_mode`-th mode ov Vt.
        """
        # SVD for _mode == 0
        # ==================
        # Consider a block sparse tensor X of shape (l,e,r).
        # We want to compute an SVD-like decomposition X = U @ S @[0] Vt such that the sparsity pattern is preserved.
        #
        # This means that:
        #     - U is a block-diagonal, orthogonal matrix.
        #     - The contraction U @[0] X does not modifying the sparsity structure
        #     - S is a diagonal matrix
        #     - The 0-(1,2)-matrification of Vt is orthogonal.
        #     - Vt is a block sparse tensor with the same sparsity structure as X.
        #       Equivalently, the 0-(1,2)-matrification of Vt has the same sparsity structure as the matrification of X.
        #
        # Assume that X contains non-zero blocks at the 3D-slices ((:a), scl_11[k], slc_12[k]) for k=1,...,K 
        # and ((a:), slc_21[l], scl_22[l]) for l=1,...,L. After a 1-(2,3)-matricisation we obtain a matrix 
        #    ┌       ┐
        #    │ X[:a] │
        #    │ X[a:] │
        #    └       ┘
        # of shape (l, e*r) and the slices take the form ((:a), scl_1[k]) and ((a:), slc_2[l]) where slc_1 and scl_2
        # are not proper slices but index arrays that select the non-zero columns of this matricisation.
        # Let X[:a] = UₐSₐVtₐ and m[a:] = UᵃSᵃVtᵃ. Then such a decomposition is given by
        #    ┌       ┐ ┌       ┐ ┌     ┐
        #    │ Uₐ    │ │ Sₐ    │ │ Vtₐ │
        #    │    Uᵃ │ │    Sᵃ │ │ Vtᵃ │
        #    └       ┘ └       ┘ └     ┘
        # It is easy to see that X = U @ S @[0] Vt and that U, S and Vt satisfy the first four properties.
        # To see this a permutation matrix Pₐ that sorts the the columns of X[:a] such that X[:a] Pₐ = [ 0 Y ], perform 
        # the SVD Y = Uʸ Sʸ Vtʸ and observe that X[:a] = Uʸ Sʸ [ 0 Vtʸ ] Ptₐ. Since Uʸ Sʸ is block-diagonal it preserves
        # the sparisity structure of [ 0 Vtʸ ] Ptₐ which has to be the same as the one of X[:a]. Since [ 0 Vtʸ ] Ptₐ is 
        # orthogonal we know that [ 0 Vtʸ ] Ptₐ = Vtₐ by the uniqueness of the SVD.
        # A similar argument holds true for X[a:] which proves the equivalent formulation of the fourth property in 
        # terms of the matrification of X.
        #
        # Note that this prove is constructive and provides a performant and numerically stable way to compute the SVD.
        #TODO: This can be done more efficiently.

        mSlices = sorted({(block[_mode].start, block[_mode].stop) for block in self.blocks})  #NOTE: slices are not hashable.

        # Check if the block structure can be retained.
        # It is necessary that there are no slices in the matricisation that are necessarily zero due to the block structure.
        assert mSlices[0][0] == 0, f"Hole found in mode {_mode}: (0:{mSlices[0][0]})"
        for j in range(len(mSlices)-1):
            assert mSlices[j][1] == mSlices[j+1][0], f"Hole found in mode {_mode}: ({mSlices[j][1]}:{mSlices[j+1][0]})"
        assert mSlices[-1][1] == self.shape[_mode], f"Hole found in mode {_mode}: ({mSlices[-1][1]}:{self.shape[_mode]})"
        # After matricisation the SVD is performed for each row-slice individually.
        # To ensure that the block structure is maintained the non-zero columns must outnumber the non-zero rows.
        for slc in mSlices:
            rows = slc[1]-slc[0]
            cols = sum(Block(blk).size for blk in self.blocks if blk[_mode].start == slc[0])     #NOTE: For coherent blocks blk[0].start == slc[0] implies equality of the slice.
            assert cols % rows == 0
            cols //= rows  # cols is the number of all non-zero columns of the `slc`-slice of the matricisation.
            assert rows <= cols, f"The {_mode}-matrification has too few non-zero columns (shape: {(rows, cols)}) for slice ({slc[0]}:{slc[1]})."  # of components[{m}][{reason[0].start}:{reason[0].stop}] has too few non-zero columns (rows: {reason[1][0]}, columns: {reason[1][1]})"

        def notMode(_tuple):
            return _tuple[:_mode] + _tuple[_mode+1:]

        # Store the blocks of the `_mode`-matrification (interpreted as a BlockSparseTensor).
        indices = np.arange(np.product(notMode(self.shape))).reshape(notMode(self.shape))
        mBlocks = []
        for slc in mSlices:
            idcs = [indices[notMode(blk)].reshape(-1) for blk in self.blocks if blk[_mode].start == slc[0]]
            idcs = np.sort(np.concatenate(idcs))
            mBlocks.append((slice(*slc), idcs))

        matricisation = np.moveaxis(self.toarray(), _mode, 0)
        mShape = matricisation.shape
        matricisation = matricisation.reshape(self.shape[_mode], -1)

        # Compute the row-block-wise SVD.
        U_blocks, S_blocks, Vt_blocks = [], [], []
        for block in mBlocks:
            u,s,vt = np.linalg.svd(matricisation[block], full_matrices=False)
            assert u.shape[0] == u.shape[1]  #TODO: Handle the case that a singular value is zero.
            U_blocks.append(u)
            S_blocks.append(s)
            Vt_blocks.append(vt)
        U = block_diag(U_blocks, format='bsr')
        S = diags([np.concatenate(S_blocks)], [0], format='dia')
        Vt = np.zeros(matricisation.shape)
        for block, Vt_block in zip(mBlocks, Vt_blocks):
            Vt[block] = Vt_block

        # Reshape Vt back into the original tensor shape.
        Vt = np.moveaxis(Vt.reshape(mShape), 0, _mode)
        #TODO: Is this equivalent to Vt = BlockSparseTensor(data, self.blocks, self.shape).toarray()?

        return U, S, Vt

    def toarray(self):
        ret = np.zeros(self.shape)
        a = 0
        for block in self.blocks:
            o = a + Block(block).size
            ret[block].reshape(-1)[:] = self.data[a:o]
            a = o
        return ret

    @classmethod
    def fromarray(cls, _array, _blocks):
        test = np.array(_array, copy=True)
        for block in _blocks:
            test[block] = 0
        assert np.all(test == 0), f"Block structure and sparsity pattern do not match."
        data = np.concatenate([_array[block].reshape(-1) for block in _blocks])
        return BlockSparseTensor(data, _blocks, _array.shape)


class BlockSparseTT(object):
    def __init__(self, _components, _blocks):
        """
        _components : list of ndarrays of order 3
            The list of component tensors for the TTTensor.
        _blocks : list of list of triples
            For the k-th component tensor _blocks[k] contains the list of its blocks of non-zero values:
                _blocks[k] --- list of non-zero blocks in the k-th component tensor
            Each block is represented by a triple of integers and slices:
                block = (slice(0,3), slice(0,4), slice(1,5))
                componentTensor[block] == componentTensor[0:3, 0:4, 1:5]
            To obtain the block this triple the slice in the component tensor:
                _blocks[k][l] --- The l-th non-zero block for the k-th component tensor.
                                  The coordinates are given by _components[k][_blocks[k][l]].

        NOTE: Later we can remove _components and augment each triple in _blocks by an array that contains the data in this block.
        """
        assert all(cmp.ndim == 3 for cmp in _components)
        assert _components[0].shape[0] == 1
        assert all(cmp1.shape[2] == cmp2.shape[0] for cmp1,cmp2 in zip(_components[:-1], _components[1:]))
        assert _components[-1].shape[2] == 1
        self.components = _components

        assert isinstance(_blocks, list) and len(_blocks) == self.order

        for m, (comp, compBlocks) in enumerate(zip(self.components, _blocks)):
            BlockSparseTensor.fromarray(comp, compBlocks)

        self.blocks = _blocks

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

        if _direction == 'left':
            assert 0 < self.corePosition

            CORE = BlockSparseTensor.fromarray(self.components[self.corePosition], self.blocks[self.corePosition])
            U, S, Vt = CORE.svd(0)

            nextCore = self.components[self.corePosition-1]
            self.components[self.corePosition-1] = (nextCore.reshape(-1, nextCore.shape[2]) @ U @ S).reshape(nextCore.shape)
            self.components[self.corePosition] = Vt

            self.__corePosition -= 1
        else:
            assert self.corePosition < self.order-1

            CORE = BlockSparseTensor.fromarray(self.components[self.corePosition], self.blocks[self.corePosition])
            U, S, Vt = CORE.svd(2)

            nextCore = self.components[self.corePosition+1]
            self.components[self.corePosition] = Vt
            self.components[self.corePosition+1] = (S @ U.T @ nextCore.reshape(nextCore.shape[0], -1)).reshape(nextCore.shape)

            self.__corePosition += 1
        self.verify()

    def dofs(self):
        return sum(BlockSparseTensor.fromarray(comp, blks).dofs() for comp, blks in zip(self.components, self.blocks))

    @classmethod
    def random(cls, _dimensions, _ranks, _blocks):
        assert len(_ranks)+1 == len(_dimensions)
        ranks = [1] + _ranks + [1]
        components = [np.zeros((leftRank, dimension, rightRank)) for leftRank, dimension, rightRank in zip(ranks[:-1], _dimensions, ranks[1:])]
        for comp, compBlocks in zip(components, _blocks):
            for block in compBlocks:
                comp[block] = np.random.randn(*comp[block].shape)
        return cls(components, _blocks)
