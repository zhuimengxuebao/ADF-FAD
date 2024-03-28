# coding:utf-8
import os

os.environ["TF_KERAS"] = '1'
os.environ['OMP_NUM_THREADS'] = "2"  # set the cpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import numpy as np

from models.Encoder import Encoder
from utils.Encoder_model_config import Encoder_get_config

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import torch


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device".format(device))
    tim_seq = 24
    batch_size = 1548  # 一次性全部运行
    # timeseq = 257
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers:线程数
    print('Using {} dataloader workers every process'.format(nw))

    # 1.加载训练集|测试集|验证集
    NYU_data_test = np.load('/ADHD200Data/NYU_dataset/NYU/NYU_data.npy')  
    len_NYU_data = len(NYU_data_test)
    print("NYU_data_test.shape:", NYU_data_test.shape) 
    t = int(len_NYU_data // tim_seq)  # 一共划分的样本数:  总数据长度 / timeseq
    NYU_data = np.expand_dims(NYU_data_test, axis=1)
    NYU_data = np.reshape(NYU_data, (t, tim_seq, NYU_data_test.shape[1]))
    NYU_data = torch.from_numpy(NYU_data)  # 将数据变成tensor
    train_num = len(NYU_data)
    print("using {} data for predicting".format(train_num))
    Train_loader = DataLoader(NYU_data, batch_size=batch_size, shuffle=False,
                              num_workers=nw)
    # create model
    Encoder_config = Encoder_get_config()  # 生成配置参数
    model = Encoder(Encoder_config)
    model.to(device)
    # 打印模型以及各层参数
    # print(model)
    # summary(model, (timeseq, NI_data.shape[2]), batch_size)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    weights_path = "/ADHD200Data/NYU_dataset/5_fold/Display/SavePth_Modeling/NYU_encoder.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))
    model.to(device)

    # 打印模型的状态字典
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # 打印优化器的状态字典
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    model.eval()
    with torch.no_grad():
        test_bar = tqdm(Train_loader)
        for t_data in test_bar:
            outputs = model(t_data.to(device))
        print("outputs.shape", outputs.shape)
        y_predict = torch.reshape(outputs, (-1, outputs.shape[2]))
        print("y_predict.shape", y_predict.shape)
        y_predict = y_predict.cpu().numpy()
    # 只用训练集训练的模型保存的所有数据预测后的位置
    np.save('/ADHD200Data/NYU_dataset/5_fold/NYU/y_all_predict.npy', y_predict)


if __name__ == '__main__':
    main()
