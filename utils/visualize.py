import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot3d(data, id_range: list = None, data_range: list = None, fig_size=(6.4, 4.8)):
    # 绘制三维图，x轴为时间，y轴为电池编号，z轴为电压
    x = range(data.shape[0])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    count = 0
    color = "#00BE00"
    id_range = range(id_range[0], id_range[1]) if id_range else range(data.shape[1])
    data_range = data_range if data_range else [0, data.shape[0]]

    for id in id_range:
        # ax.plot(x, id, z[], c='r', marker='o')
        id_str = 'cell' + str(id)
        ax.plot(x[75000:95000], data[id].values[75000:95000], zs=count, zdir='y', c=color, linewidth=1, alpha=0.8)
        ax.plot(x[data_range[0]:data_range[1]], data[data_range[0]:data_range[1], id],
                zs=count, zdir='y', c=color, linewidth=0.5, alpha=0.8)
        # 将颜色逐渐加深，使十六进制加一
        # color = '#' + str(hex(int(color[1:], 16) + 10))[2:]
        count += 1

    # 设置z轴的范围
    # ax.set_zlim(3, 4.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


def plot2d(data, id_range: list = None, data_range: list = None, fig_size=(6.4, 4.8)):
    # 绘制二维图，x轴为时间，y轴为电压
    x = range(data.shape[0])
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot()

    # color = "#00BE00"
    id_range = range(id_range[0], id_range[1]) if id_range else range(data.shape[1])
    data_range = data_range if data_range else [0, data.shape[0]]
    print(id_range)
    print(data_range)
    print(data.shape)
    for id in id_range:
        # ax.scatter(x[data_range[0]:data_range[1]], data[data_range[0]:data_range[1], id], c=color, s=0.1, alpha=0.8, label=id)
        # ax.plot(x[data_range[0]:data_range[1]], data[data_range[0]:data_range[1], id], c=color, linewidth=0.5, alpha=0.8)
        ax.plot(x[data_range[0]:data_range[1]], data[data_range[0]:data_range[1], id], linewidth=0.5, alpha=0.8, label=id)
        # 将颜色逐渐加深，使十六进制加一
        # color = '#' + str(hex(int(color[1:], 16) + 10))[2:]

    # 设置z轴的范围
    # ax.set_ylim(3, 4.5)
    # ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.legend()
    plt.show()