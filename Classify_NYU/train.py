# coding:utf-8
import os

os.environ["TF_KERAS"] = '1'
os.environ['OMP_NUM_THREADS'] = "2"  # set the cpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import numpy as np
import torch
from Classify_NYU.model_FC import CNN
# from Classify_NYU.model_FC import FC
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import time
import torch
import torch.nn as nn
from torch.utils import data
import logging


# 备注：train_dataset、 val_dataset--->代表(216,11326)刚加载进来的数据
#       train_data 、val_data---->代表(216,11325)被划分后的数据
#       train_loss  val_loss、 val_accuarcy 代表每个epoch 对应的结果，会被曲线记录下来
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


def load_data(train_dataset, val_dataset):
    # 1.读取数据
    train_label = train_dataset[:, -1]
    train_data = train_dataset[:, :-1]
    val_label = val_dataset[:, -1]
    val_data = val_dataset[:, :-1]

    # 2.数据转成tensor类型： 交叉熵损失函数中标签要转成long,数据要转成float类型
    train_label = torch.tensor(train_label).long()
    train_data = torch.tensor(train_data).float()

    val_label = torch.tensor(val_label).long()
    val_data = torch.tensor(val_data).float()

    # 3.数据封装
    dataset_train = data.TensorDataset(train_data, train_label)
    train_num = len(dataset_train)
    train_loader = data.DataLoader(dataset=dataset_train, batch_size=batchsize, shuffle=False)

    dataset_val = data.TensorDataset(val_data, val_label)
    val_num = len(dataset_val)
    val_loader = data.DataLoader(dataset=dataset_val, batch_size=batchsize, shuffle=False)

    return train_loader, val_loader, train_num, val_num


def train():
    # 1.加载数据：
    train_loader, val_loader, train_num, val_num = load_data(train_dataset, val_dataset)
    # 2. 记录log日志
    logger = get_logger(root_path + 'Data_classify/train_val_test/result/log/classifyTrainVal.log')
    # 3.记录loss和acc图像
    '/storage/fhw/STAAE_LSTM3layers_RT/Cross_NYU/Data_classify/train_val_test/result/graph/'
    save_path_graph = root_path + 'Data_classify/train_val_test/result/graph/'
    tb_writer = SummaryWriter(log_dir=save_path_graph)
    # 训练
    save_path = root_path + 'Data_classify/train_val_test/result/SavePath_Weight/FC.pth'
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        t1 = time.perf_counter()  # 统计训练一个epoch所消耗的时间
        train_steps = len(train_loader)  # 训练一个epoch所需要的步数= 总的训练集总数/batch_size
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data, label.to(device)
            data = torch.unsqueeze(data, 1)  # data:(4,1,11325)
            data = data.permute(0, 2, 1)  # data:(4,11325,1)
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_function(outputs, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("one epoch waste time:", time.perf_counter() - t1)

        model.eval()
        valid_loss = 0.0
        acc = 0.0
        with torch.no_grad():
            val_steps = len(val_loader)  # 训练一个epoch所需要的步数= 总的训练集总数/batch_size
            for (data, label) in val_loader:
                data, label = data, label.to(device)  # data:(4,11325)
                data = torch.unsqueeze(data, 1)  # data:(4,1,11325)
                data = data.permute(0, 2, 1)  # data:(4,11325,1)
                data = data.to(device)
                outputs = model(data)
                # print("outputs", outputs)
                loss = loss_function(outputs, label)
                valid_loss += loss.item()
                # print("valid_loss", valid_loss)
                pred = torch.max(outputs, dim=1)[1]
                # print("pred", pred)
                # print("label", label)
                # print("before_acc", acc)
                acc += torch.eq(pred, label).sum().item()
                # print("after_acc", acc)
            accurate_val = acc / val_num
            if accurate_val > best_acc:
                best_acc = accurate_val
                torch.save(model.state_dict(), save_path)
            train_loss = running_loss / train_steps

            finalVal_loss = valid_loss / val_steps,
            val_loss = finalVal_loss[0]

            logger.info('Epoch:[{}/{}] | train_loss{:.5f}  val_loss{:.5f} accurate_val{:5f}'.format(epoch + 1, epochs,
                                                                                                    train_loss,
                                                                                                    val_loss,
                                                                                                    accurate_val))

            # 把两条曲线放到一个图上 ||# 注意是add_scalars，不是add_scalar
            tb_writer.add_scalars('loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)
            # 把两条曲线放到各自单独的图上
            tags = ["train_loss", "val_loss", "accurate_val"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], val_loss, epoch)
            tb_writer.add_scalar(tags[2], accurate_val, epoch)


if __name__ == '__main__':
    batchsize = 2
    epochs = 120
    root_path = '/storage/fhw/STAAE_LSTM3layers_RT/Cross_NYU/'
    train_dataset = np.load(root_path + 'Data_classify/train_val_test/dataset/train_data.npy')
    val_dataset = np.load(root_path + 'Data_classify/train_val_test/dataset/val_data.npy')
    device = torch.device("cuda")
    model = CNN().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=0.0002)
    # optimizer = optim.SGD(model.parameters(), lr=0.0002, momentum=0.9)
    optimizer = optim.SGD(model.parameters(), lr=0.0002, momentum=0.9)
    loss_function = nn.CrossEntropyLoss()
    train()
