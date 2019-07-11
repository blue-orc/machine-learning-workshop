#%%
import torch
import numpy as np
import random
from torch.autograd import Variable
from torch.nn import functional as F
import matplotlib.pyplot as plt

sample1 = np.random.uniform(low=0, high=20, size=(50,))

intercept1 = random.randint(0,10)

#Calculate Y values
for w in sample1:
   Val = 2*(sample1) + intercept1


#for z in sample1:
   #print(z)
#print("Y values")
#for k in Val:
   #print (k)

plt.scatter(sample1, Val)

#Time to generate point near our line randomly
testRand = random.uniform(Val-10, Val+10)
testRand2 = np.random.uniform(Val-10, Val+10)

testRand3 = []
for k in range (1000000):
   for i in Val:
       sample = np.random.uniform(i-10, i+10)
       testRand3.append(sample)
plotX = []
for w in range (1000000):
   for h in sample1:
       sampleX = h
       plotX.append(sampleX)

#print(testRand)
#print(testRand2)
#print("This is Rand 3")
#print(len(testRand3))
#print(testRand3)
#print("This is repeat of x values")
#print(len(plotX))
#print(plotX)

#Makes Scatterplot of my data
#plt.scatter(sample1, testRand)
#plt.scatter(sample1, testRand2)
#plt.scatter(sample1, testRand3)
#plt.scatter(plotX, testRand3)

#Tutorial
x_data = Variable(torch.Tensor(plotX))
y_data = Variable(torch.Tensor(testRand3))

class LinearRegression(torch.nn.Module):
   def __init__(self):
       super(LinearRegression, self).__init__()
       self.linear = torch.nn.Linear(1, 1)
   def forward(self, x):
       y_pred = self.linear(x)
       return y_pred

print("running torch")
#%%
model = LinearRegression()
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(20000):
   model.train()
   optimizer.zero_grad()
   y_pred = model(x_data.unsqueeze(1))
   loss = criterion(y_pred, y_data.unsqueeze(1))
   loss.backward()
   optimizer.step()

new_x = Variable(torch.Tensor([[4.0]]))
y_pred = model(new_x)
print("predicted Y value: ", y_pred.data[0][0])