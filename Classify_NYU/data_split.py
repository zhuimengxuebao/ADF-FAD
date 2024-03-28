import numpy as np

# 本代码实现的功能：将数据216个样本的(功能连接体，标签)数据划分训练集(128)、验证集(44)、测试集(44)
root_path = '/storage/fhw/STAAE_LSTM3layers_RT/Cross_NYU/'
data_All = np.load(root_path + 'Data_classify/DATA_ALL/connectivity_lable_All.npy')
print("data_All.shape", data_All.shape)  # 验证一下维度是否是(216,11326)
count = 0
train_data = []
val_data = []
test_data = []
for i in data_All:
    if count < 128:
        train_data.append([])
        train_data[count].append(i)
    elif 128 <= count < 172:
        val_data.append([])
        val_data[count-128].append(i)
    else:
        test_data.append([])
        test_data[count-172].append(i)
    count = count + 1
train_data = np.asarray(train_data)
train_data = np.squeeze(train_data)
print("train_data.shape", train_data.shape)
np.save(root_path + 'Data_classify/train_val_test/dataset/train_data.npy', train_data)

val_data = np.asarray(val_data)
val_data = np.squeeze(val_data)
print("val_data.shape", val_data.shape)
np.save(root_path + 'Data_classify/train_val_test/dataset/val_data.npy', val_data)

test_data = np.asarray(test_data)
test_data = np.squeeze(test_data)
print("test_data.shape", test_data.shape)
np.save(root_path + 'Data_classify/train_val_test/dataset/test_data.npy', test_data)

