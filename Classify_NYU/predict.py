# coding:utf-8
import os

os.environ["TF_KERAS"] = '1'
os.environ['OMP_NUM_THREADS'] = "2"  # set the cpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import numpy as np
from Classify_NYU.model_FC import CNN
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.optim as optim
import torch
import torch.nn as nn
import logging


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s]%(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def evaluate():
    # 1.加载测试集
    root_path = '/storage/fhw/STAAE_LSTM3layers_RT/Cross_NYU/'
    test_dataset = np.load(root_path + 'Data_classify/train_val_test/dataset/test_data.npy')
    test_lable = test_dataset[:, -1]
    test_data = test_dataset[:, :-1]

    test_lable = torch.tensor(test_lable).long()  # 将数据变成tensor
    test_data = torch.tensor(test_data).float()

    test_num = len(test_data)
    print("using {} data for predicting".format(test_num))
    dataset_test = TensorDataset(test_data, test_lable)
    Test_loader = DataLoader(dataset=dataset_test, batch_size=2, shuffle=True)

    # 2. 记录log日志
    logger = get_logger(root_path + 'Data_classify/train_val_test/result/log/classifyTest.log')


    # 打印模型以及各层参数
    weights_path = root_path + 'Data_classify/train_val_test/result/SavePath_Weight/FC.pth'
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
    acc = 0.0
    test_loss = 0.0
    with torch.no_grad():
        for (data, label) in Test_loader:
            data, label = data, label.to(device)  # data:(2,1,11325)
            print("data.shape", data.shape)
            data = torch.unsqueeze(data, 1)  # data:(2,1,11325)
            print("data_torch.unsqueeze.shape", data.shape)
            data = data.permute(0, 2, 1)  # data:(2,11325,1)
            print("data_data.permute.shape", data.shape)
            data = data.to(device)
            outputs = model(data)
            loss = loss_function(outputs, label)
            test_loss += loss.item()
            pred = torch.max(outputs, dim=1)[1]
            acc += torch.eq(pred, label).sum().item()
        accurate_val = acc / test_num
        logger.info('test_loss{:.5f}  accurate_val{:5f}'.format(test_loss, accurate_val))


if __name__ == '__main__':
    # batchsize = 2
    device = torch.device("cuda")
    model = CNN()
    optimizer = optim.SGD(model.parameters(), lr=0.0002, momentum=0.9)
    loss_function = nn.CrossEntropyLoss()
    evaluate()
