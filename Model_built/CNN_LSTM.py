from torch import nn
import torch
from torch.utils.data import Dataset
from  torch.utils.data import DataLoader
import numpy as np

class CnnLSTM(nn.Module):
    def __init__(self, out_conv_filters, conv_kernel, conv_padding, pool_size, pool_padding,
                 lstm_hidden_unit, n_features):
        super(CnnLSTM, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

        self._conv = nn.Conv1d(in_channels=n_features, out_channels=out_conv_filters, kernel_size=conv_kernel,
                               padding=conv_padding, device=device)
        self._tanh = nn.Tanh()
        self._max_pool = nn.MaxPool1d(kernel_size=pool_size)
        self._relu = nn.ReLU(inplace=True)
        self._lstm = nn.LSTM(batch_first=True, hidden_size=lstm_hidden_unit, input_size=256, num_layers=1,
                             bidirectional=False,
                             device=device)
        self.fc1 = nn.Linear(lstm_hidden_unit, int(lstm_hidden_unit / 2))
        self.fc2 = nn.Linear(int(lstm_hidden_unit / 2), int(lstm_hidden_unit / 2))
        self.fc3 = nn.Linear(int(lstm_hidden_unit / 2), 5)
        # self._linear = nn.Linear(in_features=32 * 64, out_features=5, device=device)
        # self._flatten = nn.Flatten()
        self._sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self._conv(x)
        x = self._tanh(x)
        x = self._max_pool(x)
        x = self._relu(x)
        x,_ = self._lstm(x)
        x = self._tanh(self.fc1(x[:, -1, :]))
        x = self._tanh(self.fc2(x))
        x = self.fc3(x)
        return x
        # x = self._tanh(x[0])
        # x = self._flatten(x)
        # x = self._linear(x)
        # return x
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
source_data = torch.randn(32,35,256) # batch_size,max_length,embedding_size
# source_data = source_data.permute(0,2,1) # batch_size,embedding_size,max_length
source_label = np.random.randint(0,5,(32))
torch_data = GetLoader(source_data,source_label)
batch_size = 2
datas = DataLoader(torch_data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
if __name__ == '__main__':
    # 读取数据
    # 实例化网络、定义损失函数和优化器
    # 创建LSTM()类的对象，定义损失函数和优化器
    lstm_hidden_unit = 64
    out_conv_filters = 32
    conv_kernel = 1
    n_features = 35
    model = CnnLSTM(out_conv_filters=out_conv_filters, conv_kernel=conv_kernel, conv_padding="same", pool_padding="same", pool_size=1,
            lstm_hidden_unit= lstm_hidden_unit, n_features=n_features)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # 训练模型
    epochs = 8
    for i in range(epochs):
        running_loss = 0.0
        for seq, labels in datas:
            # 清除网络先前的梯度值
            optimizer.zero_grad()
            # 实例化模型
            labels = labels.long()
            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()  # 调用backward()自动生成梯度
            optimizer.step()  # 使用optimizer.step()执行优化器，把梯度传播回每个网络
            running_loss += single_loss.item()
        train_loss = running_loss / len(datas.sampler)
    print('loss {}'.format(train_loss))
