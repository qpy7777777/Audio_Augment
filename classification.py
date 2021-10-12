import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 实现一个nun_layers层的LSTM-RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        self.c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # input应该为(batch_size,seq_len,input_szie)
        x = x.float()
        out,_ = self.lstm(x, (self.h0, self.c0))
        out = self.fc(out[:, -1, :])
        return out


class GetLoader(Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.data)

# 随机生成数据，大小
source_data = torch.rand(8, 4, 6)
# source_data = source_data.view(-1,source_data.shape[0],source_data[1])
source_label = np.random.randint(0, 2, (32))
torch_data = GetLoader(source_data, source_label)
# 读取数据
batch_size = 2
datas = DataLoader(torch_data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
input_size = 6
hidden_size = 10
num_layers = 1
output_size = 2

model = RNN(input_size, hidden_size, num_layers, output_size)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001,momentum=0.9)

# 训练模型
EPOCH = 12
for epoch in range(EPOCH):
    running_loss = 0.0
    for step,input_data in enumerate(datas,0):
        data,label = input_data
        # 清除网络先前的梯度值
        label = label.long()
        y_pred = model(data)
        single_loss = loss_function(y_pred, label)
        # 清除网络先前的梯度值
        optimizer.zero_grad()
        single_loss.backward()  # 调用backward()自动生成梯度
        optimizer.step()  # 使用optimizer.step()执行优化器，把梯度传播回每个网络
        running_loss += single_loss.item()
        # print(running_loss)
        # 计算平均损失
    train_loss = running_loss / len(datas.sampler)
    print('epoch {}, loss {}'.format(epoch,train_loss))
        # if step % 4 == 3:  # print every 4 mini_batches,3,because of index from 0 on
        #     print('[%d,%5d]loss:%.3f' % (epoch + 1, step + 1, running_loss / 4))
        #     running_loss = 0.0

print('Finished Training')

