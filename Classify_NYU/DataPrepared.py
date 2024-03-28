import numpy as np
from nilearn.connectome import ConnectivityMeasure
# 本代码的功能：将功能连接体数据和标签数据连接起来，形成------->(功能连接体,标签)

def main():
    root_path = '/storage/fhw/STAAE_LSTM3layers_RT/Cross_NYU/'
    all_time_series = np.load(
        "/storage/fhw/STAAE_LSTM3layers_RT/ADHD200Data/NYU_dataset/5_fold/Display/visiable/STAAE_LSTM3layers_RT"
        "/visiable_epoch1200_lr0.0001/all_time_series.npy")
    print("all_time_series.shape", all_time_series.shape)  # (216,172,150)

    connectivity = ConnectivityMeasure(kind='correlation', vectorize=True)
    connectivity_biomarkers = connectivity.fit_transform(all_time_series)

    # connectivity_biomarkers.shape: (216, 11325)
    print("connectivity_biomarkers.shape:", connectivity_biomarkers.shape)
    print("connectivity_biomarkers", connectivity_biomarkers)
    # 所有人的标签
    NYU_label = np.load('/storage/fhw/STAAE/ADHD200Data/NYU_dataset/Y_all_labels.npy', allow_pickle=True).item()
    # dict.items():以列表的形式返回由字典的键值对组成的元组
    # dict_items([(1050345, 0), (1050975, 0)])
    NYU_label = sorted(NYU_label.items(), key=lambda x: x[0])
    num = 0
    Y_labels = {}
    for i in NYU_label:
        print(i)
        Y_labels[num] = i[1]
        print(" Y_labels" + "[" + str(num) + "]", Y_labels[num])
        num = num + 1
    print(len(Y_labels))  # 216个train数据：1 if ADHD(118个), 0 if control（98个）
    print("Y_labels", Y_labels)
    index = 0
    data = []
    for i in connectivity_biomarkers:
        data.append([])  # 将每个个体的功能连接体和标签存到一个二维数组中
        print("len", len(i)) # 每个个体的功能连接体内体素个数
        # print(type(i))  # <class 'numpy.ndarray'>
        i = list(i)  # 先将每个个体的功能连接体由numpy数组转成list，方便后续操作
        # print(type(i))  # <class 'list'>
        print("Y_labels[" + str(index) + "]", Y_labels[index])  # 每个功能连接体对应的标签
        i.append(Y_labels[index])   # 为每个功能连接体添加 标签
        train_numpy = np.asarray(i)  # 将新形成的（11325特征数,1标签）存到train_numpy中
        print("len", len(train_numpy))
        print(train_numpy)
        data[index].append(train_numpy) # 将216个人的数据存到data列表中
        index = index + 1
    data = np.asarray(data)  # data ：(216,1,11326)
    data = np.squeeze(data)
    print("data.shape", data.shape)
    print("data", data)


    # data ：(216,11326)
    np.save(root_path + 'Data_classify/DATA_ALL/connectivity_lable_All.npy', data)


if __name__ == "__main__":
    main()
