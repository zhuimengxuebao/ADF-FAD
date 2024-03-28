import numpy as np

root_path = '/storage/fhw/STAAE_LSTM3layers_RT/Cross_NYU/'
data_val = np.load(root_path + 'Data_classify/train_val_test/dataset/val_data.npy')
print("data_val.shape", data_val.shape)  # 验证一下维度是否是(216,11326)
print(data_val[:, -1])
