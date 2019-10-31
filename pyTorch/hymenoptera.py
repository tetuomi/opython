import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import os
from PIL import Image
import numpy as np

import pandas as pd
from sklearn import datasets, model_selection

dirs = ['ants', 'bees']

data = []
label = []

for i, d in enumerate(dirs):
    files = os.listdir('./data/' + d)
    
    for f in files:
        img = Image.open('./data/' + d + '/' + f, 'r')
        resize_img = img.resize((128, 128))
        r,g,b = resize_img.split()
        r_resize_img = np.asarray(np.float32(r)/255.0)
        g_resize_img = np.asarray(np.float32(g)/255.0)      
        b_resize_img = np.asarray(np.float32(b)/255.0)
        rgb_resize_img = np.asarray([r_resize_img, g_resize_img, b_resize_img])
        data.append(rgb_resize_img)
        
        label.append(i)
        
# pd.DataFrame(data[0][0])

data = np.array(data, dtype='float32')
label = np.array(label,dtype='int64')

train_X, test_X, train_Y, test_Y = model_selection.train_test_split(
    data, label, test_size=0.1)

print(train_X.shape)

train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(train_Y).long()

train = TensorDataset(train_X, train_Y)

train_loader = DataLoader(train, batch_size=32,shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 畳み込み層
        self.conv1 = nn.Conv2d(3, 10, 5) #(入力件数, 出力件数, フィルタサイズ)
        self.conv2 = nn.Conv2d(10, 20, 5)
        # 全結合層
        self.fc1 = nn.Linear(20 * 29 * 29, 50) # (((128 - 5 + 1) / 2) - 5 + 1) / 2
        self.fc2 = nn.Linear(50, 2)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 20 * 29 * 29)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)
    
model = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(300):
    total_loss = 0
    for train_x, train_y in train_loader:
        train_x, train_y = Variable(train_x), Variable(train_y)
        optimizer.zero_grad()
        output = model(train_x)
        loss = criterion(output, train_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.data.item()

    if (epoch+1) %50 == 0:
        print(epoch+1, '\t',total_loss)
        
test_X = np.array(test_X,dtype='float32')
test_Y = np.array(test_Y,dtype='int64')

test_X = torch.from_numpy(test_X).float()
test_Y = torch.from_numpy(test_Y).long()

test_x, test_y = Variable(test_X), Variable(test_Y)

result = torch.max(model(test_x).data, 1)[1]
accuracy = sum(test_y.data.numpy() == result.numpy()) / len(test_y.data.numpy())

print(accuracy)


'''
50  	 0.01993605640018359
100 	 0.0032639635901432484
150 	 0.0012837328686146066
200 	 0.0006236898443603422
250 	 0.00033938605338335037
300 	 0.00020556464914989192
accuracy -> 0.625
'''
