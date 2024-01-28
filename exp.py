import math
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset

# from model.AE import Model
# from model.AE_CNN import Model
# from model.AE_LSTM import Model
# from model.AE_CrossAttn import Model
# from model.AE_TemporalAttn import Model
from model.AE_Main import Model
# from model.AE_Mask import Model, MaskedMSELoss
# from model.AE_TCN import Model
# from model.AE_decomp import Model
from utils.tools import load_model, save_model
from utils.visualize import plot2d, plot3d

np.random.seed(37)
torch.manual_seed(37)


def read_data(s, e):
    data = []
    for i in range(s, e+1):
        data.append(pd.read_csv(f'data/Pack_{i}.csv'))
    # 把两个数据按列拼接
    data = pd.concat(data, axis=1)
    return data

data_train = read_data(1, 5).values
data_vali = read_data(7, 8).values
data_test = read_data(9, 9).values
print(data_test.shape)

scale = MinMaxScaler(feature_range=(-1, 1))
data_train = scale.fit_transform(data_train.reshape(-1, 7))        # 归一化
data_train = torch.from_numpy(data_train.astype(np.float32)).reshape(-1, 5*7)
data_vali = torch.from_numpy(scale.transform(data_vali.reshape(-1, 7)).astype(np.float32)).reshape(-1, 2*7)
data_test = torch.from_numpy(scale.transform(data_test).astype(np.float32))


class MyDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data.cuda() if torch.cuda.is_available() else data
        self.seq_len = seq_len

    def __getitem__(self, index):
        x = self.data[index:index+self.seq_len, :]
        # s, e = index*self.seq_len, (index+1)*self.seq_len
        # x = self.data[s:e, :]
        return x

    def __len__(self):
        # return math.floor(len(self.data)/self.seq_len)
        return len(self.data) - self.seq_len + 1


seq_len, hid_size = 96, 28  # LSTM
batch_size = 128
epochs = 100
loss_min = np.inf
early_stop, earlystop_patience = 0, 4
start_epoch, start_step = 0, 0


""" 定义模型 """
# model = Model(seq_len, hid_size, individual_channel=True)
# model = Model(input_size=7, hid_size=28)    # AE_LSTM
# model = Model(seq_len=96, hid_len=64, input_size=7, hid_size=28)    # CrossAttn
# model = Model(seq_len=96, hid_len=64, input_size=7, hid_size=28)    # TemporalAttn
model = Model(seq_len=96, hid_len=64, input_size=7, hid_size=7, kernel_size=3, block_num=3)    # AE_Main
mask_flag, mask_ratio = False, 0.2
# model = Model(seq_len=96, hid_size=28, mask_ratio=mask_ratio)    # AE_Mask
# model = Model(input_size=7, hid_size=14)    # AE_TCN
# model = Model(individual=False)    # AE_decomp
# model = Model(individual_channel=False)    # AE_vanilla
model_file = 'AE_Main_k3_blk3'

model = model.to('cuda') if torch.cuda.is_available() else model
loss_func = nn.MSELoss().to('cuda') if torch.cuda.is_available() else nn.MSELoss()
# loss_func = MaskedMSELoss().to('cuda') if torch.cuda.is_available() else MaskedMSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 如果存在权重文件，则加载权重
if os.path.exists('model.pth'):
    start_epoch, start_step, loss_min, early_stop = load_model(model, optimizer=optimizer, path=model_file+'.pth')
    print('loading model...\nloss_min:{}'.format(loss_min))





# dataloader
# s, e = 300001, 700001
s, e = 0, 200001    # 73631A6T
# dataset_train = MyDataset(data_train[s:e], seq_len)
dataset_train = ConcatDataset([MyDataset(data_train[s:e, i*7:(i+1)*7], seq_len) for i in range(data_train.shape[1]//7)])
dataset_vali = ConcatDataset([MyDataset(data_vali[s:e, i*7:(i+1)*7], seq_len) for i in range(data_vali.shape[1]//7)])
dataset_test = MyDataset(data_test[s:e], seq_len)
dataloder_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataloder_vali = DataLoader(dataset_vali, batch_size=batch_size, shuffle=False, drop_last=True)
dataloder_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=True)

# plot2d(np.array(data_train), fig_size=(9,4))
# plot2d(np.array(data_vali), fig_size=(9,4))
# plot2d(np.array(data_test), fig_size=(9,4))


for epoch in range(start_epoch, epochs):
    loss_sum, loss_count = 0, 0
    t0 = time.time()
    for step, x in enumerate(dataloder_train):
        if step < start_step:
            continue
        start_step = 0
        if mask_flag:
            out, mask = model((x, mask_flag))
            loss_ = loss_func(out, x, mask)
        else:
            out = model(x)
            loss_ = loss_func(out, x)
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()
        # break
    with torch.no_grad():
        for step, x in enumerate(dataloder_vali):
            out = model(x)
            loss_ = loss_func(out, x)
            loss_sum += loss_.item()
            loss_count += 1
            # break
    loss = loss_sum/loss_count
    t1 = time.time()
    # 损失显示小数点后6位
    print('epoch:{}/{}, loss:{}, loss_min:{}, time:{}'.format(epoch, epochs, round(loss, 6), round(loss_min, 6), round(t1-t0, 2)))
    print(loss, loss_sum, loss_count)
    if loss < loss_min:
        early_stop = 0
        loss_min = loss
        save_model(model, epoch, optimizer=optimizer, min_loss=loss_min, early_stop=early_stop, path=model_file+'.pth')
        print('save model. Early stop count:{}/{}\n'.format(early_stop, earlystop_patience))
    else:
        early_stop += 1
        print('Early stop count:{}/{}\n'.format(early_stop, earlystop_patience))
        if early_stop >= earlystop_patience:
            print('early stop')
            break


result, label = [], []
loss_sum, loss_count = 0, 0
with torch.no_grad():
    for step, x in enumerate(dataloder_test):
        # x = x.cuda() if torch.cuda.is_available() else x
        out = model(x)
        loss_ = loss_func(out, x)
        loss_sum += loss_.item()
        loss_count += 1
        result.append(out.detach().cpu().numpy())
        label.append(x.detach().cpu().numpy())
loss = loss_sum/loss_count
print('test loss:{}'.format(loss))


""" 用于可视化故障区域 """
residual = np.array(label).reshape(-1, 96, 7) - np.array(result).reshape(-1, 96, 7)
print(residual.shape)   # (5120, 96, 7)
residual_sum = np.sum(np.abs(residual), axis=1)  # 对第一个维度求和
print(residual_sum.shape)   # (5120, 7)
# 找出第一个维度中大于阈值（1）的值的索引，并将这些索引保存为列表
index = np.where(residual_sum > 1)  # tuple(array, array)


x = np.array(label).reshape(-1, 7)
y = np.array(result).reshape(-1, 7)
print(x.shape, y.shape)

plt.figure(figsize=(9, 4))
plt.plot(x[:, 3], label='label', color='r', alpha=1)
plt.plot(y[:, 3], label='predict', color='b', alpha=0.2)
plt.title(model_file)
plt.legend()
plt.tight_layout()
plt.savefig('fig/{}_reconstruct.png'.format(model_file))
plt.show()

plt.figure(figsize=(9, 4))
for i in range(7):
    plt.plot(abs(x[:,i]-y[:,i]), label=i, alpha=1)
plt.title(model_file)
plt.legend()
plt.tight_layout()
plt.savefig('fig/{}_residual.png'.format(model_file))
plt.show()

