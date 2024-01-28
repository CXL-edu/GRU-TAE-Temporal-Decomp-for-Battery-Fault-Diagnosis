import os
import time
import numpy as np
import pandas as pd

from utils.visualize import plot2d, plot3d

def get_data():
    """
    Get data from the data folder
    """
    data_path = os.path.join(os.getcwd(), 'data')
    train_path = os.path.join(data_path, 'train.csv')
    test_path = os.path.join(data_path, 'test.csv')
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data


if __name__ == "__main__":
    # data = pd.read_csv('../data/【】7N3I662O.csv')  # 【】73631A6T
    # # 打印全部列
    # # pd.set_option('display.max_columns', None)
    # # print(data.head())
    # col_name = ['collectiontime', 'vehicledata_vehiclestatus', 'vehicledata_chargestatus', 'vehicledata_sumcurrent',
    #             'chargedevicevoltagelist_voltagelist_singlebatteyvoltage']
    # data = data[col_name]
    # # 把所有数据按第一列排序
    # t0 = time.time()
    # data = data.sort_values(by='collectiontime')
    # t1 = time.time()
    # print('sort time: ', t1 - t0)
    #
    # # 把最后一列数据展开成cell1 到 cell 96，首先删除开头的[，然后删除结尾的]
    # data['chargedevicevoltagelist_voltagelist_singlebatteyvoltage'] = data['chargedevicevoltagelist_voltagelist_singlebatteyvoltage'].str.strip('[')
    # data['chargedevicevoltagelist_voltagelist_singlebatteyvoltage'] = data['chargedevicevoltagelist_voltagelist_singlebatteyvoltage'].str.strip(']')
    # data = data.join(data['chargedevicevoltagelist_voltagelist_singlebatteyvoltage'].str.split(',', expand=True).add_prefix('cell'))
    #
    #
    # t0 = time.time()
    # print(data.shape)
    # # 删除cell0 到 cell 95中大于10的数据行
    # for id in range(0, 96):
    #     id = 'cell' + str(id)
    #     data[id] = data[id].astype(float)   # 将cell0 到 cell 95的数据转换成float类型
    #     # data = data[2 < data[id] < 10]
    #     data = data[(data[id] > 2) & (data[id] < 10)]
    # t1 = time.time()
    # print('delete time: ', t1 - t0)
    # print(data.shape)
    #
    # data.to_csv('../fault7N__.csv', index=False)









    # import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    # data = pd.read_csv('data/fault73__.csv')
    # data = data.iloc[:, 5:]
    # data_k1 = pd.read_csv('data/faultK1__.csv')
    # data_k1 = data_k1.iloc[:, 5:].values
    # # plot2d(data.values, id_range=[5, 96+5], data_range=None, fig_size=(9, 4.8))
    # # plot2d(data.values, id_range=[90, 93], data_range=[150000,350001], fig_size=(9, 4.8))
    #
    # np.random.seed(37)
    #
    #
    # data_train = data.iloc[:, :6 * 7].values  # 5个机器人作为训练集
    # data_vali = data.iloc[:, 6 * 7:8 * 7].values  # 2个机器人作为验证集
    # data_test = data.iloc[:, 58:65].values  # 选择73631A6T中的7个电池
    # # print(data_test.shape)
    #
    # scale = MinMaxScaler(feature_range=(-1, 1))
    # data_train = scale.fit_transform(data_train.reshape(-1, 7))  # 归一化
    # data_train = data_train.astype(np.float32).reshape(-1, 6 * 7)[:200001, :]
    # data_vali = scale.transform(data_vali.reshape(-1, 7).astype(np.float32)).reshape(-1, 2 * 7)[:200001, :]
    # data_test = scale.transform(data_test).astype(np.float32)[:200001, :]
    # data_K1 = scale.transform(data_k1[150000:350001, 89:96]).astype(np.float32)



    # for i in range(6):
    #     print(i)
    #     # temp = data.iloc[:200001, i*7:(i+1)*7]
    #     temp = pd.DataFrame(data_train[:, i*7:(i+1)*7], columns=['cell'+str(j) for j in range(7)])
    #     # temp.columns = ['cell'+str(j) for j in range(7)]
    #     temp.to_csv('data/No_{}.csv'.format(i+1), index=False)
    #     plot2d(temp.values, id_range=[0, 7], data_range=None, fig_size=(9, 4.8))
    #
    #
    # for i in range(2):
    #     print(i)
    #     temp = pd.DataFrame(data_vali[:, i*7:(i+1)*7], columns=['cell'+str(j) for j in range(7)])
    #     temp.to_csv('data/No_{}_vali.csv'.format(i+6+1), index=False)
    #     plot2d(temp.values, id_range=[0, 7], data_range=None, fig_size=(9, 4.8))
    #
    # temp = pd.DataFrame(data_test, columns=['cell'+str(j) for j in range(7)])
    # # temp.to_csv('data/No_{}_test.csv'.format(8+1), index=False)
    # plot2d(temp.values, id_range=[0, 7], data_range=None, fig_size=(9, 4.8))
    #
    # temp = pd.DataFrame(data_K1, columns=['cell' + str(j) for j in range(7)])
    # temp.to_csv('data/No_{}_test.csv'.format(9 + 1), index=False)
    # plot2d(temp.values, id_range=[0, 7], data_range=None, fig_size=(9, 4.8))




















