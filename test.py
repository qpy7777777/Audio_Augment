# 数据准备和模型构建
# 数据准备的主要工作 1、训练集和测试集的划分；2、训练数据的归一化；3、规范输入数据的格式
# 模型构建部分主要工作
# 1、构建网络层、前向传播forward()
from torch.utils.data import Dataset
from  torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size,batch_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        # 创建LSTM层和Linear层，LSTM层提取特征，Linear层用作最后的预测
        # LSTM算法接受三个输入：先前的隐藏状态，先前的单元状态和当前输入
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,dropout=0.2)
        self.fc1 = nn.Linear(hidden_size,int(hidden_size/2))
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(int(hidden_size/2),int(hidden_size/2))
        self.fc3 = nn.Linear(int(hidden_size/2),output_size)
        # 初始化隐含状态及细胞状态C，hidden_cell变量包含先前的隐藏转态和单元状态
#         self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
#                             torch.zeros(1, 1, self.hidden_layer_size))
        self.h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        self.c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
    # forward该函数的参数一般为输入数据，返回值是输出数据。
    def forward(self,x):
        # lstm的输出是当前时间步的隐藏状态ht和单元状态ct以及输出lstm_out
#         lstm_out, self.hidden_cell = self.lstm(x.view(len(x), 1, -1), self.hidden_cell)
        x = x.float()
        lstm_out,_ = self.lstm(x,(self.h0,self.c0))
        # 按照lstm的格式修改input_seq的形状，作为linear层的输入
#         predictions = self.linear(lstm_out.view(len(input_seq), -1))
#         out = self.relu(self.fc1(lstm_out[:,-1,:]))
#         out = self.relu(self.fc1(lstm_out.contiguous().view(x.shape[1],-1)))
        out = self.relu(self.fc1(lstm_out[:, -1, :]))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# 加载自己的数据集
# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(Dataset):
    def __init__(self,data_root,data_label):
        self.data = data_root
        self.label = data_label
    def __getitem__(self,index):
        data = self.data[index]
        label = self.label[index]
        return data,label
    def __len__(self):
        return len(self.data)

# 随机生成数据，大小
source_data = np.arange(1,33).reshape(4,2,4)
# source_data = source_data.view(-1,source_data.shape[0],source_data[1])
source_label = np.random.randint(0,2,(4))
torch_data = GetLoader(source_data,source_label)
# 读取数据
batch_size = 2
datas = DataLoader(torch_data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
# 实例化网络、定义损失函数和优化器
# 创建LSTM()类的对象，定义损失函数和优化器
input_size = source_data.shape[2]
hidden_size = 20
num_layers= 1
output_size = 2
model = LSTM(input_size,hidden_size,num_layers,output_size,batch_size)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)

# 训练模型
epochs = 8
for i in range(epochs):
    running_loss = 0.0
    for seq,labels in datas:
        #清除网络先前的梯度值
        optimizer.zero_grad()
        #初始化隐藏层数据
#         model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
#                              torch.zeros(1, 1, model.hidden_layer_size))
        #实例化模型
        labels = labels.long()
        y_pred = model(seq)
        #计算损失，反向传播梯度以及更新模型参数
        #训练过程中，正向传播生成网络的输出，计算输出和实际值之间的损失值
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()#调用backward()自动生成梯度
        optimizer.step()#使用optimizer.step()执行优化器，把梯度传播回每个网络
        running_loss += single_loss.item()
    train_loss = running_loss / len(datas.sampler)
    print('loss {}'.format(train_loss))

