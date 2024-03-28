# coding:utf-8
import os

os.environ["TF_KERAS"] = '1'
os.environ['OMP_NUM_THREADS'] = "2"  # set the cpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import numpy as np
import torch

# 利用mask将2D序列转为4D数据
from nilearn.input_data import NiftiMasker
from nilearn.image import iter_img
from nilearn.plotting import plot_prob_atlas, plot_stat_map, plot_matrix, plot_connectome
from nilearn.plotting import find_probabilistic_atlas_cut_coords, show
import nibabel as nib
from nilearn.regions import RegionExtractor
from nilearn.connectome import ConnectivityMeasure
from sklearn.linear_model import Lasso
import matplotlib.pyplot  as plt
import scipy

mask_img = 'ADHD200_mask_152_4mm.nii'
masker = NiftiMasker(mask_img=mask_img,
                     standardize=True,
                     detrend=1,
                     smoothing_fwhm=6.)

masker.fit()


def thresholding(components):
    print("components.shape\n", components.shape) 
    S = np.sqrt(np.sum(components ** 2, axis=1)) 
    print("S.shape\n", S.shape)
    print("S--before\n", S)
    S[S == 0] = 1
    print("S--after\n", S)
    components /= S[:, np.newaxis]
    print("len(components)\n", len(components)) 

    # Flip signs in each composant so that positive part is l1 larger
    # than negative part. Empirically this yield more positive looking maps
    # than with setting the max to be positive.
    for component in components:
        if np.sum(component > 0) < np.sum(component < 0):
            component *= -1
    return components


def plot_net(components, root_path):
    # 2D-->4D Transform the 2D data matrix back to an image in brain space.
    components_img = masker.inverse_transform(components)
    print("inverse_transform(components):components_img", components_img)

    # 为了Dice_3d中OR的计算
    nib.save(components_img, root_path + 'visiable_epoch1200_lr0.0001/components_img.nii')

    # Plot all ICA components together:  output_file :要将绘图导出到的图像文件的名称。有效的扩展名是 .png、.pdf、.svg。
    plot_prob_atlas(components_img, title='All components',
                    output_file=root_path + "visiable_epoch1200_lr0.0001/All_components.png")
    
    # 三个方向的切片信息
    for i, cur_img in enumerate(iter_img(components_img)):
        components_img_path = root_path + 'ortho/'
        outname = components_img_path + str(i) + '_rsn.png'
        plot_stat_map(cur_img, display_mode="ortho", title="%d" % i,
                      colorbar=True, black_bg=True, annotate=True, output_file=outname)

    # RSNs网络 ：显示64个组件
    for i, cur_img in enumerate(iter_img(components_img)):
        components_img_path = root_path + 'RSNs/'
        outname = components_img_path + str(i) + '_rsn.png'
        plot_stat_map(cur_img, display_mode="z", title="%d" % i, cut_coords=10,
                      black_bg=True, annotate=True, colorbar=True, output_file=outname)
    return components_img

# Extract regions from networks
def ExtractRegions(components_img, num_components, root_path):
    extractor = RegionExtractor(components_img, threshold=2.3,
                                thresholding_strategy='ratio_n_voxels',
                                extractor='local_regions',
                                standardize=True, min_region_size=1350)
    # Just call fit() to process for regions extraction
    extractor.fit()
    # Extracted regions are stored in regions_img_
    regions_extracted_img = extractor.regions_img_
    # Each region index is stored in index_
    regions_index = extractor.index_
    # Total number of regions extracted

    n_regions_extracted = regions_extracted_img.shape[-1]

    # Visualization of region extraction results
    title = ('%d regions are extracted from %d components.'
             '\nEach separate color of region indicates extracted region'
             % (n_regions_extracted, num_components))
    # 在默认的MNI模板下将概率图集绘制到解剖图像上
    plot_prob_atlas(regions_extracted_img, view_type='filled_contours', title=title,
                    output_file=root_path + "/visiable_epoch1200_lr0.0001/regions_extracted_img.png")
    return extractor, n_regions_extracted, regions_extracted_img


def plot_matrices(matrices, matrix_kind):
    n_matrices = len(matrices)  # 4 显示前4个功能连接体
    fig = plt.figure(figsize=(n_matrices * 4, 4))
    for n_subject, matrix in enumerate(matrices):
        plt.subplot(4, n_matrices, n_subject + 1)
        matrix = matrix.copy()
        np.fill_diagonal(matrix, 0)
        vmax = np.max(np.abs(matrix))
        title = '{},suject{}'.format(matrix_kind, n_subject)
        plot_matrix(matrix, vmin=-vmax, vmax=vmax, cmap='RdBu_r', auto_fit=True,
                    title=title, figure=fig, colorbar=True)
        show()


def ComputeCorrelationCoefficients(extractor, n_regions_extracted, root_path):
    # First we need to do subjects timeseries signals extraction and then estimating
    # correlation matrices on those signals.
    # To extract timeseries signals, we call transform() from RegionExtractor object
    # onto each subject functional data stored in func_filenames.
    # To estimate correlation matrices we import connectome utilities from nilearn
    # 先引入原始数据和标签
    dir_name_2 = '/ADHD200Data/NYU_dataset/NYU/'
    func_filenames = []
    for x in os.listdir(dir_name_2):
        dir = dir_name_2 + str(x)
        if os.path.isdir(dir):
            file = dir + '/sfnwmrda' + str(x) + '_session_1_rest_1.nii.gz'
            func_filenames.append(file)
        else:
            print(dir)
    func_filenames = sorted(func_filenames)
    print(func_filenames)
    print(len(func_filenames))

    # 1.加载训练集标签 ！！！！
    # 重要语法：# items()适用取出字典的元素  ||  item() 适用tensor和numpy：取出里面的元素
    import numpy as np
    NYU_label = np.load('/ADHD200Data/NYU_dataset/Y_all_labels.npy', allow_pickle=True).item()
    # dict.items():以列表的形式返回由字典的键值对组成的元组
    NYU_label = sorted(NYU_label.items(), key=lambda x: x[0])
    Y_labels = []
    for i in NYU_label:
        print(i[1])
        Y_labels.append(i[1])
    print(len(Y_labels)) 

    adhd_time_series = []
    all_time_series = []
    adhd_labels = []
    filename_count = 0
    for filename, is_adhd in zip(func_filenames, Y_labels):
        filename_count += 1
        print("filename" + str(filename_count), filename)
        # 从RegionExtractor对象调用transform来提取时间序列信号
        timeseries_each_subject = extractor.transform(filename)
        print("timeseries_each_subject.shape:", timeseries_each_subject.shape) 
        all_time_series.append(timeseries_each_subject)
        if is_adhd == 1: 
            adhd_time_series.append(timeseries_each_subject)
    print('Data has {0} ADHD.'.format(len(adhd_time_series))) 

    all_time_series = np.asarray(all_time_series) 
    print("all_time_series", all_time_series.shape)  
    # 将numpy数组保存成npy格式
    np.save(root_path + "visiable_epoch1200_lr0.0001/all_time_series.npy", all_time_series)

    # 以下是效果展示，展示ADHD患者的提取的各个脑区间的根据提取的时间序列计算的相关性

    # ConnectivityMeasure ：一个计算不同类型功能连接体矩阵的类
    # vectorize:如果为True，则连接矩阵被重塑为一维数组，只返回其扁平的下三角形部分。默认= False。
    conn_measure = ConnectivityMeasure(kind='correlation')
    # fit_transform : 对每个受试者的时间序列计算相关性
    # 从ADHD的ROI时间序列列表中，用conn_measure计算各个相关矩阵
    connectivity_biomarkers = conn_measure.fit_transform(adhd_time_series)
    print("adhd_time_series_connectivity_biomarkers.shape:", connectivity_biomarkers.shape)
    # For each kind, all individual coefficients are stacked in a unique 2D matrix.
    print('{} correlation biomarkers for each subject.'.format(
        connectivity_biomarkers.shape[1])) 

    import numpy as np
    # 拟合所有患者的平均相关性
    mean_correlations = np.mean(connectivity_biomarkers, axis=0).reshape(n_regions_extracted, n_regions_extracted)
    print("mean_correlations.shape", mean_correlations.shape)  
    return connectivity_biomarkers, mean_correlations


def PlotResultingConnectomes(mean_correlations, n_regions_extracted, regions_extracted_img, root_path):
    title = 'Correlation between %d regions' % n_regions_extracted

    # First plot the matrix
    # First plot the matrix
    plot_matrix(mean_correlations, vmax=1, vmin=-1, auto_fit=True,
                colorbar=True, title=title)
    plt.savefig(root_path + 'visiable_epoch1200_lr0.0001/mean_correlations.png')
    print("regions_extracted_img.shape", regions_extracted_img.shape)
    nib.save(regions_extracted_img,
             root_path + 'visiable_epoch1200_lr0.0001/regions_extracted_img.nii')
    regions_img = regions_extracted_img
    coords_connectome = find_probabilistic_atlas_cut_coords(regions_img)
   
    plot_connectome(mean_correlations, coords_connectome,
                    edge_threshold='99.80%', title=title,
                    colorbar=True,
                    output_file=root_path+"visiable_epoch1200_lr0.0001/mean_correlations_connectome.png")
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device".format(device))

    # 1.加载预测后的数据和原始数据
    y_predict = np.load('/ADHD200Data/NYU_dataset/5_fold/NYU/y_all_predict.npy') 
    print("y_predict.shape", y_predict.shape)
    NYU_data = np.load('/ADHD200Data/NYU_dataset/NYU/NYU_data.npy') 
    clf = Lasso(alpha=0.05)
    clf.fit(y_predict, NYU_data)
    print("after clf.coef_.T.shape\n", clf.coef_.T.shape)
    print("after clf.coef_.T", clf.coef_.T)

    components = thresholding(clf.coef_.T) 
    print("after thresholding components.shape\n", components.shape)  
    print("after thresholding components", components)

    scipy.stats.mstats.zscore(components, axis=1)
    print("after zscore components.shape\n", components.shape) 
    print("after zscore components", components)

    components[np.absolute(components) < 0.01] = 0
    print("after np.absolute(components) < 0.01, components.shape\n", components.shape)  # (28546,64)
    print("afternp.absolute(components) < 0.01, components", components)

    components[components < 0] = 0
    print("after components < 0, components.shape\n", components.shape)  # (28546,64)
    print("after components < 0, components", components)

    root_path = '/ADHD200Data/NYU_dataset/5_fold/Display/visiable/'
    # 用于后续与ICA方法计算IOU
    np.save(
        root_path + 'visiable_epoch1200_lr0.0001/NYU_components.npy', components)

    num_components = len(components)
    print("len(components)\n", num_components)  

    # 2. Visualization of functional networks
    components_img = plot_net(components, root_path)

    # 3. Extract regions from networks
    extractor, n_regions_extracted, regions_extracted_img = ExtractRegions(components_img, num_components, root_path)

    # 4. Compute correlation coefficients
    connectivity_biomarkers, mean_correlations = ComputeCorrelationCoefficients(extractor, n_regions_extracted, root_path)
    # 5. Plot resulting connectomes
    PlotResultingConnectomes(mean_correlations, n_regions_extracted, regions_extracted_img, root_path)


if __name__ == '__main__':
    main()
