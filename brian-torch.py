#%%
import torch
import numpy as np
import random
from torch.autograd import Variable
from torch.nn import functional as F
import matplotlib.pyplot as plt

np.random.seed(12)
num_observations = 50000

x_data = []
y_data = []
for x in range (10000):
    set1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
    x_data.append(set1.flatten())
    y_data.append(0)
    set2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)
    x_data.append(set2.flatten())
    y_data.append(1)
    if x % 1000 == 0:
        print(x)


#Tutorial
x_tensor = Variable(torch.Tensor(x_data))
y_tensor = Variable(torch.Tensor(y_data))

class LinearRegression(torch.nn.Module):
   def __init__(self):
       super(LinearRegression, self).__init__()
       self.linear = torch.nn.Linear(num_observations*2, 1)
   def forward(self, x):
       y_pred = self.linear(x)
       return y_pred

print("running torch")
#%%
model = LinearRegression()
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(2000):
   model.train()
   optimizer.zero_grad()
   y_pred = model(x_tensor)
   loss = criterion(y_pred, y_tensor.unsqueeze(1))
   loss.backward()
   optimizer.step()

new_x = Variable(torch.Tensor([[4.0]]))
y_pred = model(new_x)
print("predicted Y value: ", y_pred.data[0][0])