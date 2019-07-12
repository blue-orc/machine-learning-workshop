#%%
import torch
import numpy as np
import random
from torch.autograd import Variable
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

def train(model,x_tensor,y_tensor):
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.train()
    optimizer.zero_grad()
    y_pred = model(x_tensor)
    loss = criterion(y_pred, y_tensor.unsqueeze(1))
    loss.backward()
    optimizer.step()





#class LinearRegression(torch.nn.Module):
#   def __init__(self):
#       super(LinearRegression, self).__init__()
#       self.linear = torch.nn.Linear(num_observations*2, 1)
#   def forward(self, x):
#       y_pred = self.linear(x)
#       return y_pred

class LogisticRegression(torch.nn.Module):    
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(num_observations*2, 1)
    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred
        



#%%
#model = LogisticRegression()


#for epoch in range(2000):
#   model.train()
#   optimizer.zero_grad()
#   y_pred = model(x_tensor)
#   loss = criterion(y_pred, y_tensor.unsqueeze(1))
#   loss.backward()
#   optimizer.step()

if __name__ == '__main__':
    np.random.seed(12)
    num_observations = 50000
    x_data = []
    y_data = []
    for x in range (5000):
        set1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
        x_data.append(set1.flatten())
        y_data.append(0)
        set2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)
        x_data.append(set2.flatten())
        y_data.append(1)
        if x % 1000 == 0:
            print(x)
    print("finished generating data")

    #Tutorial
    x_tensor = Variable(torch.Tensor(x_data))
    y_tensor = Variable(torch.Tensor(y_data))

    print("running torch")  
    num_processes = 4
    model = LogisticRegression()
    # NOTE: this is required for the ``fork`` method to work
    model.share_memory()

    processes = []
    for epoch in range(2000):
        for rank in range(num_processes):
            p = mp.Process(target=train, args=(model,x_tensor,y_tensor))
            p.start()
            print("mem: "+torch.cuda.memory_allocated())
            print("cached: " +torch.cuda.memory_cached())
            processes.append(p)
        for p in processes:
            p.join()

#new_x = Variable(torch.Tensor([[4.0]]))
#y_pred = model(new_x)
#print("predicted Y value: ", y_pred.data[0][0])

