import numpy as np
import csv
from numba import vectorize
#import matplotlib.pyplot as plt
#%matplotlib inline

def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))

def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
    return ll

@vectorize(['ndarray(np.float32, ndarray, int, float32, boolean)'], target='cuda')
def logistic_regression(features, target, num_steps, learning_rate, add_intercept = False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))
        
    weights = np.zeros(features.shape[1])
    
    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient
        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient
        
        # Print log-likelihood every so often
        if step % 1000 == 0:
            result = log_likelihood(features, target, weights))
            print('Step ' + str(step) + ': ' + str(result))
        
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


simulated_separableish_features = np.vstack((set1, set2)).astype(np.float32)
simulated_labels = np.hstack((np.zeros(num_observations),
                              np.ones(num_observations)))

weights = logistic_regression(simulated_separableish_features, simulated_labels, num_steps = 300000, learning_rate = 5e-5, add_intercept=True)
print('ok')
