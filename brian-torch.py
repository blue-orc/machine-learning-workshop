#%%
import torch
import numpy as np
import random
import csv
from torch.autograd import Variable
from torch.nn import functional as F
import matplotlib.pyplot as plt

np.random.seed(12)
num_observations = 5

test1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], 5)
test2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], 5)
test1 = test1.flatten()
test2 = test2.flatten()
print(test1)

x_data = []
x_fieldnames = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10' ]
y_data = []
y_fieldname = ['y']
for x in range (2000):
    set1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
    x_data.append(set1.flatten())
    y_data.append(0)
    set2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)
    x_data.append(set2.flatten())
    y_data.append(1)
    if x % 1000 == 0:
        print(x)
print("finished generating")

#Tutorial
device = torch.device("cuda:0")

class LogisticRegression(torch.nn.Module):    
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(num_observations*2, 1)
    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred
        
model = LogisticRegression()
model = torch.nn.DataParallel(model)
model.to(device)

x_tensor = Variable(torch.Tensor(x_data)).to(device)
y_tensor = Variable(torch.Tensor(y_data)).to(device)

print("running torch")

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(2000):
   model.train()
   optimizer.zero_grad()
   y_pred = model(x_tensor)
   loss = criterion(y_pred, y_tensor.unsqueeze(1))
   loss.backward()
   optimizer.step()
#%%
test1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], 10)
test2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], 10)
test1 = test1.flatten()
test2 = test2.flatten()
test1_x = Variable(torch.Tensor(test1))
test2_x = Variable(torch.Tensor(test2))
y_pred1 = model(test1_x)
y_pred2 = model(test2_x)
print("predicted Y1 value: ", y_pred1.data)
print("predicted Y2 value: ", y_pred2.data)

#%%
