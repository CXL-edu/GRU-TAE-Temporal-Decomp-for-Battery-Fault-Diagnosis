import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from utils.tools import load_model, save_model
from utils.visualize import plot2d, plot3d
#显示全部pandas列
pd.set_option('display.max_columns', None)
# 打印全部list元素
# np.set_printoptions(threshold=np.inf)

data1 = pd.read_csv('./data/Pack_9_test.csv').values
data1 = torch.from_numpy(data1).float()
# data2 = pd.read_csv('./data/Pack_10_test.csv').values
# data2 = torch.from_numpy(data2).float()

# plot2d(data1)
# plot2d(data2)


def dynamic_threshold(data, factor=3, base_threshold=0.2):
    """
    动态阈值法
    data: (batch, seq_len, 7)
    """
    mean_cell = np.mean(np.abs(data), axis=1)  # 对第一个维度求均值, (batch, 7)
    mean, std = np.mean(mean_cell, axis=1), np.var(mean_cell, axis=1)  # (batch, ), (batch, )

    threshold = mean + factor * std
    # threshold = mean/np.power(std, 1/3)
    # 根据mean中数据的大小，设置不同的阈值
    mean_flag = np.sum(mean_cell > base_threshold, axis=1) > 0
    threshold[~mean_flag] = base_threshold
    threshold = threshold.reshape(data.shape[0],1,1)

    temp = np.sum(data > threshold, axis=1)
    index = np.where(temp > 10)  # tuple(array, array)
    # print('index: ', index)
    return index, threshold


class MyDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data.cuda() if torch.cuda.is_available() else data
        self.seq_len = seq_len

    def __getitem__(self, index):
        # x = self.data[index:index+self.seq_len, :]
        s, e = index*self.seq_len, (index+1)*self.seq_len
        x = self.data[s:e, :]
        return x

    def __len__(self):
        return math.floor(len(self.data)/self.seq_len)
        # return len(self.data) - self.seq_len + 1


seq_len, hid_size = 96, 64
# seq_len, hid_size = 96, 28  # LSTM
batch_size = 128
epochs = 100
loss_min = np.inf
early_stop, earlystop_patience = 0, 4
start_epoch, start_step = 0, 0


""" 定义模型 """
#
# model = Model(seq_len, hid_size)
# model = Model(seq_len=96, hid_len=64, input_size=7, hid_size=7, block_num=2)    # AE_Main
#
# model = Model(seq_len=96, hid_len=64, input_size=7, hid_size=28)    # TemporalAttn
# model = Model(individual=False)
# model = Model(individual_channel=False)

# model_list = []
#
from model.AE import Model
model = Model(seq_len, 28, individual_channel=False)
_, _, min_loss, _ = load_model(model, only_model=True, path='model_weight/AE_vanilla/AE_28.pth')
# model.append(model_AE)

# from model.AE_LSTM import Model
# model = Model(input_size=7, hid_size=28)    # AE_LSTM
# _, _, min_loss, _ = load_model(model, only_model=True, path='model_weight/model_AE_LSTM.pth')
# model.append(model_GRU)

# from model.AE_TCN import Model
# model = Model(input_size=7, hid_size=28)    # AE_TCN
# _, _, min_loss, _ = load_model(model, only_model=True, path='model_weight/AE_TCN/AE_TCN_28.pth')
# model.append(model_TCN)
#
# from model.AE_CrossAttn import Model
# model = Model(seq_len=96, hid_len=64, input_size=7, hid_size=96)    # CrossAttn
# _, _, min_loss, _ = load_model(model, only_model=True, path='model_weight/AE_Cross/AE_CrossAttn_hl64_hs96.pth')
# model.append(model_cross)
#
# from model.AE_Main import Model
# model = Model(seq_len=96, hid_len=64, input_size=7, hid_size=7, kernel_size=3, block_num=3)    # AE_Main
# _, _, min_loss, _ = load_model(model, only_model=True, path='model_weight/AE_Main/AE_Main_k3_blk3.pth')
# model.to('cuda') if torch.cuda.is_available() else model
# model.append(model_main)

# from model.AE_TemporalAttn import Model
# model = Model(seq_len=96, hid_len=64, input_size=7, hid_size=28)    # TemporalAttn
# _, _, min_loss, _ = load_model(model, only_model=True, path='model_weight/AE_Temporal.pth')


model_name = 'AE'


print('loading model...\n Validation loss_min:{}'.format(min_loss))

loss = nn.MSELoss().cuda() if torch.cuda.is_available() else nn.MSELoss()

# for data in [data1, data2]:
s, e = 0, -1
# s, e = 60000, 110000
dataset_test = MyDataset(data1[s:e], seq_len)
dataloder_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=True)

result, label = [], []
with torch.no_grad():
    for step, x in enumerate(dataloder_test):
        # x = x.cuda() if torch.cuda.is_available() else x
        out = model(x)
        result.append(out.detach().cpu().numpy())
        label.append(x.detach().cpu().numpy())
result = np.array(result).reshape(-1, 96, 7)    # (batch, seq_len, feature_num)
label = np.array(label).reshape(-1, 96, 7)

# loss_ = loss(torch.from_numpy(result), torch.from_numpy(label))
# print('Test loss:{}'.format(loss_))
""" 用于可视化故障区域 """
residual = np.array(label).reshape(-1, 96, 7) - np.array(result).reshape(-1, 96, 7)
residual = abs(residual)
fault_idx, threshold = dynamic_threshold(residual)
thd_x = np.arange(0, residual.shape[0])*96

# unique_elements, counts = np.unique(fault_idx[1], return_counts=True)
# count_data1, count_data2 = 0, 0
# for i in range(len(unique_elements)):
#     print("Frequency of", unique_elements[i], ":", counts[i])
#     count_data1 = count_data1 + counts[i] if unique_elements[i] != 3 else count_data1
#     count_data2 = count_data2 + counts[i] if unique_elements[i] != 2 else count_data2
# print('data1 fault count: ', count_data1)
# print('data2 fault count: ', count_data2)
# print(8*'-')




x = np.array(label).reshape(-1, 7)
y = np.array(result).reshape(-1, 7)

# reconstruction figure
plt.figure(figsize=(9, 4))
plt.plot(x[:, 3], label='Original', color='blue', alpha=0.7)
plt.plot(y[:, 3], label='Reconstruction', color='red', alpha=0.7)
# plt.title(model_file)
plt.legend(loc="best", fontsize=16)
plt.tick_params(labelsize=16)
plt.tight_layout()
plt.subplots_adjust(left=0.06, right=0.995, top=0.98, bottom=0.1)
# 对x轴使用科学计数法
# plt.ticklabel_format(axis='x', style='', scilimits=(0,0))
plt.savefig('fig1/{}_reconstruct.svg'.format(model_name))
# plt.show()

# residual 2D figure
plt.figure(figsize=(9, 4))
for i in range(7):
    plt.plot(abs(x[:,i]-y[:,i]), label='cell'+str(i), alpha=0.8)
    # print(np.sum(abs(x[:,i]-y[:,i])>0.2))
plt.plot(thd_x, threshold[:, 0, 0], label='Dynamic threshold', color='k', alpha=1, linestyle='--')
plt.tick_params(labelsize=16)
plt.legend(loc="best", fontsize=16)
plt.tight_layout()
# 设置y轴与边界的距离
plt.subplots_adjust(left=0.06, right=0.995, top=0.98, bottom=0.1)
plt.savefig('fig1/{}_residual.emf'.format(model_name))
# plt.show()

# 将残差绘制成三维图
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for i in range(7):
    # ax.plot(x, id, z[], c='r', marker='o')
    id_str = 'cell' + str(i)
    y_ = abs(x[:,i]-y[:,i])
    x_ = range(y_.shape[0])
    ax.plot(x_, y_, zs=i+1, zdir='y', linewidth=1, alpha=1, label=id_str)
    # 绘制阈值
    # ax.plot(thd_x, np.repeat(threshold[i], thd_x.shape[0]), zs=i+1, zdir='y', linewidth=1, alpha=1, label=id_str, c='k')
# 设置y轴使用科学计数法
ax.ticklabel_format(axis='x', style='', scilimits=(0,0))
# 设置图片的角度
ax.view_init(elev=14, azim=-38)

# ax.legend(loc=(10, 10, 10))
# plt.plot(thd_x, threshold.reshape(-1,1), 'o', markersize=0.1, label='Dynamic threshold', color='k', alpha=1, linestyle='--')
# 绘制threshold图，假设threshold数据为[1,2,3,2], 则绘制出的图为[1,1,2,2,3,3,2,2]，即每个threshold值对应两个点，且后一个点与前一个点重合
# 且不同区域间不相连


# plt.title(model_file)
# plt.legend(loc="best")
plt.tight_layout()
plt.savefig('fig1/{}_residual_3d.svg'.format(model_name))
# plt.show()



