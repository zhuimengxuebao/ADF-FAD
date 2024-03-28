# coding:utf-8
import os

os.environ["TF_KERAS"] = '1'
os.environ['OMP_NUM_THREADS'] = "2"  # set the cpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import time
from models.RAE_Model import RAE

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary
from utils.EarlyStopping import EarlyStopping
from utils.batchify_utils import batchify

from matplotlib import pyplot as plt

from utils.Encoder_model_config import Encoder_get_config
from utils.Decoder_model_config import Decoder_get_config
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


# #######k折划分############
def get_k_fold_data(i):
    train_data_path = '/ADHD200Data/NYU_dataset/NYU_5_fold_data/train_data/'
    test_data_path = '/ADHD200Data/NYU_dataset/NYU_5_fold_data/val_data/'

    x_train_name = 'x_train_' + str(i) + '_2D.npy'
    x_test_name = 'x_test_' + str(i) + '_2D.npy'

    x_train_path = train_data_path + x_train_name
    x_test_path = test_data_path + x_test_name

    # 加载训练集和标签(由于标签是字典类型的，所以此处还需把标签提取出来)
    x_train_data = np.load(x_train_path)
    x_test_data = np.load(x_test_path)
    return x_train_data, x_test_data


def k_fold(k, num_epochs, learning_rate, root_path, tim_seq=24):
    train_loss_list = []  # 记录训练集K折的损失
    test_loss_list = []  # 记录验证集K折的损失
    # 记录log日志
    logger = get_logger(root_path + 'extract_FBN_exp_RT.log')
    for i in range(1, k + 1):
        data = get_k_fold_data(i)  # 获取k折交叉验证的训练和验证数据

        logger.info('-' * 89)
        logger.info('第{}折----start training!----'.format(i))
        # 获得所有epoch 的loss的list列表--->(绘制早停曲线)
        train_loss, test_loss = train(*data, i, num_epochs, learning_rate, root_path, logger, tim_seq=24)

        # visualize every fold the loss as the network trained
        fig = plt.figure(figsize=(10, 8))
        plt.plot(range(1, len(train_loss) + 1), train_loss, label=str(i) + '_fold Training Loss')
        plt.plot(range(1, len(test_loss) + 1), test_loss, label=str(i) + '_fold Validation Loss')

        # find position of lowest validation loss
        minposs = test_loss.index(min(test_loss)) + 1
        plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.ylim(0, 1)  # consistent scale
        plt.xlim(0, len(train_loss) + 1)  # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        fig.savefig(root_path + str(i) + '_fold/Display/loss_plot.png', bbox_inches='tight')
        # visualize the loss as the network trained ---------above---------------------

        logger.info('第{}折----finish training!----'.format(i))
        # train_loss | test_loss 均是记录的一个折中所有epoch的loss，train_loss中最小的可能是early stopping之后的。
        # 所以valid最小，不一定对应train_loss最小。只能用valid最小的列表下标去找对应的train_loss
        # logger.info('第[{}]折：min_train_loss={:.5f}\t min_test_loss={:.5f}\t'.format(i, train_loss[np.argmin(train_loss)],
        #                                                                            test_loss[np.argmin(test_loss)]))
        logger.info('第[{}]折：min_train_loss={:.5f}\t min_test_loss={:.5f}\t'.format(i, train_loss[np.argmin(test_loss)],
                                                                                   test_loss[np.argmin(test_loss)]))
        train_loss_list.append(train_loss[np.argmin(test_loss)])
        test_loss_list.append(test_loss[np.argmin(test_loss)])

    # 分别将list列表转成numpy数组,利用numpy数组的性质可计算标准差和方差
    logger.info('所有折的 train_loss_list{}'.format(train_loss_list))
    logger.info('所有折的 test_loss_list{}'.format(test_loss_list))
    train_loss_all = np.array(train_loss_list)
    test_loss_all = np.array(test_loss_list)
    print('#' * 10, '最终k折交叉验证结果', '#' * 10)
    ####体现步骤四#####
    logger.info('average fold train loss:%.4f' % (np.sum(train_loss_all) / k))
    logger.info('average fold test loss:%.4f' % (np.sum(test_loss_all) / k))

    logger.info('variance of fold train loss:%.4f' % (np.var(train_loss_all)))
    logger.info('variance of fold test loss:%.4f' % (np.var(test_loss_all)))

    logger.info('standard deviation of fold train loss:%.4f' % (np.std(train_loss_all)))
    logger.info('standard deviation of fold test loss:%.4f' % (np.std(test_loss_all)))
    logger.info('{}折中最优模型是第{}折 |最小train_loss：{}| 最小test_loss：{}'.format(k, np.argmin(test_loss_list) + 1,
                                                                        train_loss_list[np.argmin(test_loss_list)],
                                                                        test_loss_list[np.argmin(test_loss_list)]))


def train(x_train_data, x_test_data, i, num_epochs, learning_rate,
          root_path, logger, tim_seq=24):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device".format(device))
    save_path_graph = root_path + str(i) + '_fold/Display/tensorboard_test_epoch1200_lr_0.0001_timseq172/runs' \
                                           '/brain_experiment'
    tb_writer = SummaryWriter(log_dir=save_path_graph)
    # --------------------训练集数据预处理-------------------------------------------------------
    # 数据预处理：对输入的二维数据增加中间一个维度，变成三维数据；
    print("before x_train_data.shape:", x_train_data.shape)
    x_train_data, nbatch = batchify(x_train_data, tim_seq) 
    print("after batchify x_train_data.shape:", x_train_data.shape)
    x_train_datas = np.expand_dims(x_train_data, axis=1) 
    x_train_datas = np.reshape(x_train_datas, (nbatch, tim_seq, x_train_data.shape[1]))
    x_train_datas = torch.from_numpy(x_train_datas)  # 将数据变成tensor
    train_num = len(x_train_datas)  # 1232 OR 1239
    print("using {} data for training".format(train_num))
    train_loader = DataLoader(TensorDataset(x_train_datas, x_train_datas), batch_size=2, shuffle=False)

    # --------------------测试集（验证集）数据预处理-------------------------------------------------------
    # 测试数据扩充一个维度，变成三维数据
    print("before x_test_data.shape:", x_test_data.shape) 
    x_test_data, nbatch = batchify(x_test_data, tim_seq)
    print("after batchify x_test_data.shape:", x_test_data.shape) 
    x_test_datas = np.expand_dims(x_test_data, axis=1) 
    x_test_datas = np.reshape(x_test_datas, (nbatch, tim_seq, x_test_data.shape[1])) 
    x_test_datas = torch.from_numpy(x_test_datas)  # 将数据变成tensor
    test_num = len(x_test_datas) 
    print("using {} data for testing".format(test_num))
    test_loader = DataLoader(TensorDataset(x_test_datas, x_test_datas), batch_size=2, shuffle=False)

    # 2.引入模型，并放在GPU上
    Encoder_config = Encoder_get_config()  # 生成配置参数
    Decoder_config = Decoder_get_config()
    model = RAE(Encoder_config, Decoder_config)
    model.to(device)
    # print(model)

    # 这里使用了Adam优化算法
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()  # 用于回归问题，预测值和真实值（x-y）2

    # 模型权重保存路径
    save_path_weight = root_path + str(i) + '_fold/Display/SavePth_Modeling/NYU_epoch1200_lr0.0001_timseq172.pth'
    save_encoder_weight_path = root_path + str(
        i) + '_fold/Display/SavePth_Modeling/NYU_encoder_epoch1200_lr0.0001_timseq172.pth'
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    # initialize the early_stopping object
    patience = 20  # 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
    early_stopping = EarlyStopping(patience, verbose=True)

    # 训练模型，直到 epoch == n_epochs 或者触发 early_stopping 结束训练
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0  # 记录每个epoch的损失
        train_bar = tqdm(train_loader)
        # t1 = time.perf_counter()
        for step, data in enumerate(train_bar, start=0):  # 分批训练
            x_train, x_train = data  # (1,24,28546)
            optimizer.zero_grad()
            outputs = model(x_train.to(device))
            # 计算损失函数
            loss = loss_function(outputs, x_train.to(device))
            loss.backward()
            optimizer.step()  # loss.item() 记录每一个batch的平均loss
            train_losses.append(loss.item())  # 每个epoch最终的损失

        # 每个epoch计算测试集的loss,accuracy
        # 设置模型为评估/测试模式，关闭dropout，并将模型参数锁定
        model.eval()
        # testLoss = 0.0  # 记录每个epoch的损失
        test_bar = tqdm(test_loader)
        test_steps = len(test_loader)
        # t1 = time.perf_counter()
        with torch.no_grad():
            for step, data in enumerate(test_bar, start=0):
                x_test, x_test = data  # (43,4,28546)
                outputs = model(x_test.to(device))
                # 计算损失函数
                loss = loss_function(outputs, x_test.to(device))
                valid_losses.append(loss.item())

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)  # 存储本次epoch的训练损失
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)  # 存储所有epoch的训练损失
        avg_valid_losses.append(valid_loss)

        logger.info(
            '第[{}]折：end of Epoch:[{}/{}] | time:{:5.2f}s | train_loss={:.5f} | test_loss={:.5f}'.format(i, epoch,
                                                                                                        num_epochs,
                                                                                                        (time.time() - epoch_start_time),
                                                                                                        train_loss,
                                                                                                        valid_loss))

        # 把两条曲线放到一个图上 ||# 注意是add_scalars，不是add_scalar
        tb_writer.add_scalars('loss', {'train_loss': train_loss, 'valid_loss': valid_loss}, epoch)
        # 把两条曲线放到各自单独的图上
        tags = ["train_loss", "valid_loss"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], valid_loss, epoch)

        # clear lists to track next epoch(记录每个epoch所有损失)
        train_losses = []
        valid_losses = []

        # 调用早停机制防止过拟合
        early_stopping(valid_loss, model, save_path_weight, save_encoder_weight_path)

        if early_stopping.early_stop:
            print("Early stopping")
            break  # 只是跳出本次for循环

    # mini前缀的是返回所有epoch中最小的那个；avg开头的是所有epoch
    return avg_train_losses, avg_valid_losses


def main():
    root_path = '/ADHD200Data/NYU_dataset/'
    k_fold(5, 2000, 0.0001, root_path, tim_seq=24)  # k=5,五折交叉验证


if __name__ == '__main__':
    main()
