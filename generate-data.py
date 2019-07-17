import math
import numpy as np
import csv
#import matplotlib.pyplot as plt
#%matplotlib inline

def writeCSV(filename,  fieldnames, dataset):
    with open(filename + '.csv', mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        count = 0
        while count < len(dataset):
            if count%100 == 0:
                print(filename + ': wrote ' + str(count) + ' samples')
            
            row = {}
            fieldcount = 0
            for field in fieldnames:
                row[field] = str(dataset[count][fieldcount])
                fieldcount += 1

            writer.writerow(row)
            count += 1
            
        
print("Setting up data generation...")
np.random.seed(12)
num_observations = 2000
x_fieldnames = []
y_fieldnames = ['y']
#np.random.multivariate_normal() returns pair of values, which are flattened, so num_observations must be multiplied by 2
for obs in range (num_observations*2):
    x_fieldnames.append('x' + str(obs))

x_data = []
y_data = []
print("Begin generating data...")
for x in range (5000):
    set1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
    x_data.append(set1.flatten())
    y_data.append(0)
    set2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)
    x_data.append(set2.flatten())
    y_data.append(1)
    if x % 1000 == 0:
        print("Generated " + str(x*2) + " samples")
print("Finished generating data...")

x_csv_name = "x_data"
y_csv_name = "y_data"

print("Writing X data to " + x_csv_name + ".csv")
writeCSV(x_csv_name, x_fieldnames, x_data)

print("Writing Y data to " + y_csv_name + ".csv")
writeCSV(y_csv_name, y_fieldnames, y_data)
print("Data generation complete")