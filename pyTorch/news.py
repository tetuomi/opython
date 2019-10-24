import os
import re

from sklearn import datasets, model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.tokenfilter import POSKeepFilter

import pandas as pd

import numpy as np

dirs = ['it-life-hack', 'movie-enter']

for i, d in enumerate(dirs):
    files = os.listdir('./data/text/' + d)
    
    for file in files:
        f = open('./data/text/' + d + '/' + file, 'r', encoding='utf-8')
        raw = f.read()
        reg_raw = re.sub(r'[0-9a-zA-z]', '', raw)
        reg_raw = reg_raw.replace('\n', '')
#         print(reg_raw)
                
        f.close()

x_ls = []
y_ls = []

tmp1 = []
tmp2 = ''

tokenizer = Tokenizer()
token_filters = [POSKeepFilter(['名詞'])]
analyzer = Analyzer([], tokenizer, token_filters)

for i, d in enumerate(dirs):
    files = os.listdir('./data/text/' + d)
    
    for file in files:
        f = open('./data/text/' + d + '/' + file, 'r', encoding='utf-8')
        raw = f.read()
    
        reg_raw = re.sub(r'[0-9a-zA-z]', '', raw)
        reg_raw = reg_raw.replace('\n', '')
    
        for token in analyzer.analyze(reg_raw):
            tmp1.append(token.surface)
            tmp2 = ' '.join(tmp1)

        x_ls.append(tmp2)
        tmp1 = []
    
        y_ls.append(i)
        f.close()

# pd.DataFrame(x_ls)
# print(x_ls)
# print(x_ls[0])
# print(y_ls)

x_array = np.array(x_ls)
y_array = np.array(y_ls)

cntvec = CountVectorizer()
x_cntvecs = cntvec.fit_transform(x_array)
x_cntarray = x_cntvecs.toarray()

# pd.DataFrame(x_cntarray)

# for k, v in sorted(cntvec.vocabulary_.items(), key=lambda x:x[1]):
#     print(k, v)
    
tfidf_vec = TfidfVectorizer(use_idf=True)
x_tfidf_vecs = tfidf_vec.fit_transform(x_array)
x_tfidf_array = x_tfidf_vecs.toarray()

# pd.DataFrame(x_tfidf_array)

train_X, test_X,train_Y, test_Y = model_selection.train_test_split(
    x_tfidf_array, y_array, test_size=0.2)

# print(len(train_X))
# print(len(test_X))

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(train_Y).long()

test_X = torch.from_numpy(test_X).float()
test_Y = torch.from_numpy(test_Y).long()

# print(train_X.shape)
# print(train_Y.shape)

train = TensorDataset(train_X, train_Y)
# print(train[0])
train_loader = DataLoader(train, batch_size=100, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(20869, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return F.log_softmax(x)
    
model = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# for epoch in range(1000):
#     total_loss = 0
#     for train_x, train_y in train_loader:
#         train_x, train_y = Variable(train_x), Variable(train_y)
#         optimizer.zero_grad()
#         output = model(train_x)
#         loss = criterion(output, train_y)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.data[0]
    
#     if (epoch + 1) % 100 == 0:
#         print(epoch + 1, total_loss)
        
for epoch in range(1000):
    total_loss = 0
    for train_x, train_y in train_loader:
        train_x, train_y = Variable(train_x), Variable(train_y)
        optimizer.zero_grad()
        output = model(train_x)
        loss = criterion(output, train_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.data.item()

    if (epoch+1) % 100 == 0:
        print(epoch+1, '\t',total_loss)

test_x, test_y = Variable(test_X), Variable(test_Y)
result = torch.max(model(test_x).data, 1)[1]
accuracy = sum(test_y.data.numpy() == result.numpy()) / len(test_y.data.numpy())
print(accuracy)


/*
100 	 1.19209286886246e-09
200 	 0.0 ?
300 	 0.0 ?
400 	 0.0 ? 
500 	 0.0 ?
600 	 0.0 ?
700 	 0.0 ?
800 	 0.0 ?
900 	 0.0 ?
1000 	 0.0 ?
0.9712643678160919
*/
