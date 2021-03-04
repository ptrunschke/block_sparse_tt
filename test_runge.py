import numpy as np
from numpy.polynomial.legendre import legval

from misc import random_polynomial
from als import ALS


# ==========
#  TRAINING
# ==========

bstt = random_polynomial([7,7,7,7], 7)

N = int(1e5)  # number of samples
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
