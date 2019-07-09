import math
import numpy as np
import csv
from numba import vectorize, jit, cuda, float32
#import matplotlib.pyplot as plt
#%matplotlib inline

def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
    return ll

@cuda.jit
def fast_matmul(A, B, C):
    TPB = 16
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp

def logistic_regression(features, target, num_steps, learning_rate, intercept, weights):
    
    for step in range(num_steps):
        scores = np.empty((features.shape[0],weights.shape[0]), features.dtype)
        fast_matmul(features, weights, scores)
        predictions = 1 / (1 + np.exp(-scores))

        # Update weights with gradient
        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient
        
        # Print log-likelihood every so often
        if step % 1000 == 0:
            ##result = log_likelihood(features, target, weights)
            print('ok')
            #print('Step ' + str(step) + ': ' + str(result))
        
    return weights

def writeCSV(filename, dataset):
    with open(filename + '.csv', mode='w') as csv_file:
        fieldnames = ['x', 'y']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        count = 0
        for value in x1:
            if count%10000 == 0:
                print(filename + ': wrote ' + str(count) + ' datapoints')
            
            writer.writerow({'x': value[0], 'y': value[1]})
            count += 1

np.random.seed(12)
num_observations = 50000

set1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
#writeCSV('sample1', set1)
set2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)
#writeCSV('sample2', set2)


features = np.vstack((set1, set2)).astype(np.float64)
simulated_labels = np.hstack((np.zeros(num_observations),
                              np.ones(num_observations)))

intercept = np.ones((features.shape[0], 1), np.float64)
features = np.hstack((intercept, features))
weights = np.zeros(features.shape[1])
weights = logistic_regression(features, simulated_labels, 300000, 5e-5, intercept, weights)
print('ok')
