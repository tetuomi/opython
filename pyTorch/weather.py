import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline

dat = pd.read_csv('./data/data.csv', skiprows=[0,1,2,4], encoding='shift-jis')
# dat

temp = dat['平均気温(℃)']

# temp.plot()
# plt.show()

train_x = temp[:1461] # 2011/1/1 ~ 2014/12/31
test_x = temp[1461:]  # 2015/1/1 ~ 2016/12/31

train_x = np.array(train_x)
test_x = np.array(test_x)

ATTR_SIZE = 180 # 説明変数の件数

tmp = []
train_X = []

for i in range(0, len(train_x) - ATTR_SIZE):
    tmp.append(train_x[i:i + ATTR_SIZE])

train_X = np.array(tmp)

# pd.DataFrame(train_X

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(180, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 180)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = Net()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
    total_loss = 0
    d = []
    for i in range(100):
        index = np.random.randint(0, 1281)
        d.append(train_X[index])
    d = np.array(d, dtype="float32")
    d = Variable(torch.from_numpy(d))
    
    optimizer.zero_grad()
    output = model(d)
    loss = criterion(output, d)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    
    if (epoch + 1) % 100 == 0:
        print(epoch + 1, total_loss)
        
# plt.plot(d.data[0].numpy(), label='original')
# plt.plot(output.data[0].numpy(), label='output')
# plt.legend(loc='upper right')
# plt.show()

tmp = []
test_X = []

tmp.append(test_x[:180])
tmp.append(test_x[180:360])
tmp.append(test_x[360:540])
tmp.append(test_x[540:720])
test_X = np.array(tmp, dtype='float32')

# pd.DataFrame(test_X)

d = Variable(torch.from_numpy(test_X))
output = model(d)

# plt.plot(test_X.flatten(), label='original')
# plt.plot(output.data.numpy().flatten(), label='prediction')
# plt.legend(loc='upper right')
# plt.show()

test = test_X.flatten()
pred = output.data.numpy().flatten()

total_score = []
for i in range(0, 720):
    dist = (test[i] - pred[i])
    score = pow(dist, 2)
    total_score.append(score)
    
total_score = np.array(total_score)
max_score = np.max(total_score)
total_score = total_score / max_score

# print(total_score)

plt.plot(total_score)
plt.show()
