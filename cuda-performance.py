import math  # Note that for the CUDA target, we need to use the scalar functions from the math module, not NumPy
import time
import numpy as np

SQRT_2PI = np.float32((2*math.pi)**0.5)  # Precompute this constant as a float32.  Numba will inline it at compile time.

@vectorize(['float32(float32, float32, float32)'], target='cuda')
def gaussian_pdf(x, mean, sigma):
    '''Compute the value of a Gaussian probability density function at x with given mean and sigma.'''
    return math.exp(-0.5 * ((x - mean) / sigma)**2) / (sigma * SQRT_2PI)

# Evaluate the Gaussian a million times!
x = np.random.uniform(-3, 3, size=1000000).astype(np.float32)
mean = np.float32(0.0)
sigma = np.float32(1.0)

# Quick test
gaussian_pdf(x[0], 0.0, 1.0)

import scipy.stats # for definition of gaussian distribution
norm_pdf = scipy.stats.norm

start = time.time()
norm_pdf.pdf(x, loc=mean, scale=sigma)
end = time.time()
print (end - start)

start2 = time.time()
gaussian_pdf(x, mean, sigma)
end2 = time.time()
print (end2 - start2)