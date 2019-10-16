#PyTorchライブラリの読み込み
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#Scikit-learnライブラリの読み込み
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

#Pandasライブラリの読み込み
import pandas as pd

#ワインデータセットの読み込みと表示
wine = load_wine()
#wine

#データフレーム形式で説明変数を表示
#pd.DataFrame(winedata, columns=wine.feature_names)

#目的変数の表示
#wine.target

#説明変数と目的変数を格納
wine_data = wine.data[0:130]
wine_target = wine.target[0:130]

#データセットを訓練用とテスト用に分割
train_X, test_X, train_Y, test_Y = train_test_split(wine_data, wine_target, test_size=0.2)

# print(train_X)
# print(test_X)
# print(train_Y)
# print(test_Y)

#データの長さを確認
# print(len(train_X))
# print(len(test_X))

#訓練用のテンソルの作成
train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(train_Y).long()

#テスト用のテンソル作成
test_X = torch.from_numpy(test_X).float()
test_Y = torch.from_numpy(test_Y).long()

#テンソルの件数を表示
print(train_X.shape)
print(train_Y.shape)

#説明変数と目的変数のテンソルをまとめる
train = TensorDataset(train_X, train_Y)

#1つめのテンソルを確認
print(train[0])

#ミニバッチに分ける
train_loader = DataLoader(train, batch_size=16, shuffle=True)

#ネットワークの作成
class Net(nn.Module):                 #ニューラルネットワークの基本クラス
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(13, 96)  #入力データの線形変換 -> y=ax+b 
        self.fc2 = nn.Linear(96, 2)   #入力データの線形変換 -> y=ax+b
        
    def forward(self, x):
        x = F.relu(self.fc1(x))       #ReLU関数(中間層)
        x = self.fc2(x)
        return F.log_softmax(x)       #対数ソフトマックス関数(出力層)

#インスタンスの生成
model = Net()

#誤差関数のセット
criterion = nn.CrossEntropyLoss()

#最適化関数のセット
optimizer = optim.SGD(model.parameters(), lr=0.01)

#学習開始
for epoch in range(300):
    total_loss = 0
    #分割したデータの取り出し
    for train_x, train_y in train_loader:
        #計算グラフの構築
        train_x, train_y = Variable(train_x), Variable(train_y)
        #勾配のリセット
        optimizer.zero_grad()
        #順伝播の計算
        output = model(train_x)
        #誤差の計算
        loss = criterion(output, train_y)
        #逆伝播の計算
        loss.backward()
        #重みの計算
        optimizer.step()
        #誤差の累積
        total_loss += loss.item()
    #累積誤差を50回ごとに表示
    if (epoch + 1) % 50 == 0:
        print(epoch + 1, total_loss)
        
test_x, test_y = Variable(test_X), Variable(test_Y)
result = torch.max(model(test_x).data, 1)[1]
accuracy = sum(test_y.data.numpy() == result.numpy()) / len(test_y.data.numpy())
print(accuracy)
