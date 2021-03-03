import numpy as np
from scipy.sparse import block_diag


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
        assert isinstance(_blocks, list)
        self.blocks = [Block(block) for block in _blocks]
        assert all(block.is_valid() for block in self.blocks) and sum(block.size for block in self.blocks) == self.data.size
        for i in range(len(self.blocks)):
            for j in range(i):
                assert self.blocks[i].disjoint(self.blocks[j]) and self.blocks[i].coherent(self.blocks[j])
        assert isinstance(_shape, tuple) and np.all(np.array(_shape) > 0)
        shapeBlock = Block(slice(0,dim,1) for dim in _shape)
        assert all(shapeBlock.contains(block) for block in self.blocks)
        self.shape = _shape

    # TODO: I think this method is unnecessary when assert_stable_svd() is used.
    # def is_holey(self):
    #     """
    #     Checks if there are slices in the matrix that have to be zero by the block structure.
    #     These holes mean that there can be no SVD that preserves the sparsity structure.
    #     """
    #     #NOTE: This can be done more efficiently.
    #     #NOTE: The condition that the middle slices are not holey may restrict TT-Blocks unnecessarily.
    #     masks = [np.full((dim,), False, dtype=bool) for dim in self.shape]
    #     for block in self.blocks:
    #         for mask,slc in zip(masks, block):
    #             mask[slc] = True
    #     return any(not np.all(mask) for mask in masks)

    def assert_stable_svd(self, _mode):
        """
        Check if an SVD along the _mode-th mode will retain the the block structure.

        For this it is important that there are no slices in the matrix that have to be zero by the block structure.
        These holes mean that there can be no SVD that preserves the sparsity structure.
        The SVD will be considered for a matricisation of the tensor.
        The considered matricisation has `_mode` as its rows and all other modes as its columns.

        Parameters
        ----------
        _mode: non-negative int
            The mode that is chosen as one side of the matricisation.

        Returns
        -------
        tuple or None
            Returns None if the SVD will retain the block structure.
            Otherwise return the first `_mode`-slice that will loose its structure
            and the size of the non-zero part of the sliced matrification.

        """
        modeSlices = sorted({(block[_mode].start, block[_mode].stop) for block in self.blocks})  #NOTE: slices are not hashable.
        for j in range(len(modeSlices)-1):
            assert modeSlices[j][1] == modeSlices[j+1][0], f"Hole found in mode {_mode}: ({modeSlices[j][1]}:{modeSlices[j+1][0]})"
        for slc in modeSlices:
            rows = slc[1]-slc[0]
            cols = sum(Block(blk).size for blk in self.blocks if blk[_mode].start == slc[0])     #NOTE: For coherent blocks blk[0].start == slc[0] implies equality of the slice.
            cols /= rows  # cols is the number of all non-zero columns of the `slc`-slice of the matricisation.
            assert rows <= cols, f"The {_mode}-matrification has too few non-zero columns (shape: {(rows, cols)}) for slice ({slc[0]}:{slc[1]})."  # of components[{m}][{reason[0].start}:{reason[0].stop}] has too few non-zero columns (rows: {reason[1][0]}, columns: {reason[1][1]})"

    def to_ndarray(self):
        ret = np.zeros(self.shape)
        a = 0
        for block in self.blocks:
            o = a + Block(block).size
            ret[block].reshape(-1)[:] = self.data[a:o]
            a = o
        return ret


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
            data = np.concatenate([comp[block].reshape(-1) for block in compBlocks])
            comp = BlockSparseTensor(data, compBlocks, comp.shape)
            comp.assert_stable_svd(0)
            comp.assert_stable_svd(2)

        #TODO: I do not think that this is relevant. Matrix-matrix multiplication works block-wise --- independet of how the blocks are distributed.
        # leftSlices = [ sorted({(block[0].start, block[0].stop) for block in cblocks}) for cblocks in _blocks ]   #NOTE: slices are not hashable.
        # rightSlices = [ sorted({(block[2].start, block[2].stop) for block in cblocks}) for cblocks in _blocks ]  #NOTE: slices are not hashable.
        # #NOTE: Check that for every two neighboring components every block in the left component has a corresponding block in the right component.
        # #      Otherwise, if for example the right component would be dense, the left component must be dense as well after a core move to the left.
        # assert all(rightSlices[m] == leftSlices[m+1] for m in range(self.order-1))

        self.blocks = _blocks

        # TODO: Gib BlockSparseTensor diese Methode und der Klasse einfach auch eine svd-methode!
        def dense_slices(_blocks, _shape, _mode):
            """
            For an SVD with `_mode == 0` the tensor is reshaped into shape (tensor.shape[0], -1).
            dense_slices()[k] is used to select the columns of this matrification for the k-th slice in the rows.
            Similarly, for an SVD with `_mode == 2` the tensor is reshaped into shape (-1, tensor.shape[2]).
            Then dense_slices()[k] is used to select the rows of this matrification for the k-th slice in the columns.
            """
            rowSlices = sorted({(block[_mode].start, block[_mode].stop) for block in _blocks})  #NOTE: slices are not hashable.
            def notMode(_tuple): return _tuple[:_mode] + _tuple[_mode+1:]
            indices = np.arange(np.product(notMode(_shape))).reshape(notMode(_shape))
            ret = []
            for rowSlc in rowSlices:
                colIndices = [indices[notMode(blk)].reshape(-1) for blk in _blocks if blk[_mode].start == rowSlc[0]]
                colIndices = np.sort(np.concatenate(colIndices))
                ret.append((slice(*rowSlc), colIndices))
            return ret

        # leftBlocks and rightBlocks contain the slices of non-zero blocks in the matrifications needed for the core move.
        self.leftBlocks = [dense_slices(blocks, comp.shape, 0) for blocks, comp in zip(self.blocks, self.components)]
        self.rightBlocks = [[tpl[::-1] for tpl in dense_slices(blocks, comp.shape, 2)] for blocks, comp in zip(self.blocks, self.components)]
        #TODO: Remove the unnecessary reordering in self.rightBlocks!

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
            # test = np.einsum('ler,rds', newCore, oldCore)

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

            #TODO: This is done in self.verify().
            # oldCore_test = np.array(oldCore)
            # for blk in self.blocks[self.corePosition]:
            #     oldCore_test[blk] = 0
            # assert np.allclose(oldCore_test, 0)

            # assert np.allclose(test, np.einsum('ler,rds', newCore, oldCore))

            self.components[self.corePosition-1] = newCore
            self.components[self.corePosition] = oldCore
            self.__corePosition -= 1
        else:
            assert self.corePosition < self.order-1
            oldCore = self.components[self.corePosition]
            newCore = self.components[self.corePosition+1]
            # test = np.einsum('ler,rds', oldCore, newCore)

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

            #TODO: This is done in self.verify().
            # oldCore_test = np.array(oldCore)
            # for blk in self.blocks[self.corePosition]:
            #     oldCore_test[blk] = 0
            # assert np.allclose(oldCore_test, 0)

            # assert np.allclose(test, np.einsum('ler,rds', oldCore, newCore))

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
