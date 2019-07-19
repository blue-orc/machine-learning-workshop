#%%
import torch
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
import cx_Oracle

db = cx_Oracle.connect(user="ADMIN", password="Oracle12345!", dsn="mlwadw_high")
print("Connected to Oracle ADW")

def getData(db):
    cur = db.cursor()
    statement = "SELECT * FROM (SELECT X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, Y FROM SAMPLE_DATA) WHERE ROWNUM <= 1000000"
    cur.execute(statement)
    res = cur.fetchall()
    npRes = np.array(res)
    x_data = npRes[:, :10]
    y_data = npRes[:,10]
    return x_data, y_data

print("Selecting sample data from ADW")
x_data, y_data = getData(db)

print("Loading data and model on to GPU")
device = torch.device("cuda:0")

class LogisticRegression(torch.nn.Module):    
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(len(x_data), 1)
    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred
        
model = LogisticRegression()
model = torch.nn.DataParallel(model)
model.to(device)

x_tensor = Variable(torch.Tensor(x_data)).to(device)
y_tensor = Variable(torch.Tensor(y_data)).to(device)

print("Running model training")

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

num_epochs = 2000
for epoch in range(num_epochs):
    pctComplete = epoch / num_epochs * 100
    print ("{:.2f}".format(pctComplete)+"%", end="\r")
    model.train()
    optimizer.zero_grad()
    y_pred = model(x_tensor)
    loss = criterion(y_pred, y_tensor.unsqueeze(1))
    loss.backward()
    optimizer.step()

print("Training complete")

print("Running Prediction using model")
#%%
test1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], 10)
test2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], 10)
test1 = test1.flatten()
test2 = test2.flatten()
test1_x = Variable(torch.Tensor(test1))
test2_x = Variable(torch.Tensor(test2))
y_pred1 = model(test1_x)
y_pred2 = model(test2_x)
print("Predicted Y1 value: ", y_pred1.data[0], " Expected: 0")
print("Predicted Y2 value: ", y_pred2.data[0], " Expected: 1")

#%%
