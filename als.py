#NOTE: This implementation is not meant to be memory efficient or fast but rather to test the approximation capabilities of the proposed model class.
import numpy as np
from bstt import Block, BlockSparseTensor


class ALS(object):
    def __init__(self, _bstt, _measurements, _values, _verbosity=0):
        self.bstt = _bstt
        assert isinstance(_measurements, np.ndarray) and isinstance(_values, np.ndarray)
        assert len(_measurements) == self.bstt.order
        assert all(compMeas.shape == (len(_values), dim) for compMeas, dim in zip(_measurements, self.bstt.dimensions))
        self.measurements = _measurements
        self.values = _values
        self.verbosity = _verbosity
        self.maxSweeps = 100
        self.targetResidual = 1e-8
        self.minDecrease = 1e-4

        self.leftStack = [np.ones((len(self.values),1))] + [None]*(self.bstt.order-1)
        self.rightStack = [np.ones((len(self.values),1))]
        self.bstt.assume_corePosition(self.bstt.order-1)
        while self.bstt.corePosition > 0:
            self.move_core('left')

    def move_core(self, _direction):
        self.bstt.move_core(_direction)
        if _direction == 'left':
            if self.verbosity >= 2: print(f"move_core: {self.bstt.corePosition+1} --> {self.bstt.corePosition}")
            self.leftStack.pop()
            self.rightStack.append(np.einsum('ler, ne, nr -> nl', self.bstt.components[self.bstt.corePosition+1], self.measurements[self.bstt.corePosition+1], self.rightStack[-1]))
        elif _direction == 'right':
            if self.verbosity >= 2: print(f"move_core: {self.bstt.corePosition-1} --> {self.bstt.corePosition}")
            self.rightStack.pop()
            self.leftStack.append(np.einsum('nl, ne, ler -> nr', self.leftStack[-1], self.measurements[self.bstt.corePosition-1], self.bstt.components[self.bstt.corePosition-1]))
        else:
            raise ValueError(f"Unknown _direction. Expected 'left' or 'right' but got '{_direction}'")

    def residual(self):
        core = self.bstt.components[self.bstt.corePosition]
        L = self.leftStack[-1]
        E = self.measurements[self.bstt.corePosition]
        R = self.rightStack[-1]
        pred = np.einsum('ler,nl,ne,nr -> n', core, L, E, R)
        return np.linalg.norm(pred -  self.values) / np.linalg.norm(self.values)

    def microstep(self):
        core = self.bstt.components[self.bstt.corePosition]
        L = self.leftStack[-1]
        E = self.measurements[self.bstt.corePosition]
        R = self.rightStack[-1]
        coreBlocks = self.bstt.blocks[self.bstt.corePosition]

        Op_blocks = []
        Param_blocks = []
        for block in coreBlocks:
            op = np.einsum('nl,ne,nr -> nler', L[:, block[0]], E[:, block[1]], R[:, block[2]])
            op.shape = len(self.values), Block(block).size
            Op_blocks.append(op)
            Param_blocks.append(core[block].reshape(-1))
        Op = np.concatenate(Op_blocks, axis=1)
        Param = np.concatenate(Param_blocks)
        Res = np.linalg.solve(Op.T @ Op, Op.T @ self.values)
        core[...] = BlockSparseTensor(Res, coreBlocks, core.shape).toarray()

    def run(self):
        prev_residual = self.residual()
        if self.verbosity >= 1: print(f"Initial residuum: {prev_residual:.2e}")
        for sweep in range(self.maxSweeps):
            while self.bstt.corePosition < self.bstt.order-1:
                self.microstep()
                self.move_core('right')
            while self.bstt.corePosition > 0:
                self.microstep()
                self.move_core('left')

            residual = self.residual()
            if self.verbosity >= 1: print(f"[{sweep}] Residuum: {residual:.2e}")

            if residual < self.targetResidual:
                if self.verbosity >= 1:
                    print(f"Terminating (targetResidual reached)")
                    print(f"Final residuum: {self.residual():.2e}")
                return

            if residual > prev_residual:
                if self.verbosity >= 1:
                    print(f"Terminating (residual increases)")
                    print(f"Final residuum: {self.residual():.2e}")
                return

            if (prev_residual - residual) < self.minDecrease*residual:
                if self.verbosity >= 1:
                    print(f"Terminating (minDecrease reached)")
                    print(f"Final residuum: {self.residual():.2e}")
                return

            prev_residual = residual

        if self.verbosity >= 1: print(f"Terminating (maxSweeps reached)")
        if self.verbosity >= 1: print(f"Final residuum: {self.residual():.2e}")
