# coding:utf-8
import os

os.environ["TF_KERAS"] = '1'
os.environ['OMP_NUM_THREADS'] = "2"  # set the cpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from nilearn._utils.niimg_conversions import _resolve_globbing
from nilearn.input_data import NiftiMasker
from nilearn.decomposition.base import mask_and_reduce


def prepare_filenames(dir_name):
    func_filenames = []
    for x in os.listdir(dir_name):
        file = dir_name + str(x) + '/sfnwmrda' + str(x) + '_session_1_rest_1.nii.gz'
        if os.path.isfile(file):
            func_filenames.append(file)
        else:
            print("missing " + file)
    func_filenames = sorted(func_filenames)
    # list of 4D nifti files for each subject
    return func_filenames


# data :4D->2D
def prepare_data(func_filenames):
    imgs = _resolve_globbing(func_filenames)
    mask_img = 'ADHD200_mask_152_4mm.nii'
    masker = NiftiMasker(mask_img=mask_img,
                         standardize=True,
                         detrend=1,
                         smoothing_fwhm=6.,
                         memory="/NYU_dataset/nilearn_cache",
                         memory_level=1)
    # fmri_masked = masker.fit()
    masker.fit()
    for index, func_filename in enumerate(func_filenames):
        fmri_masked = masker.transform(func_filename)
        if index == 0:
            iterator = fmri_masked
        else:
            iterator = np.concatenate((iterator, fmri_masked), axis=0)
    print("fmri_masked.shape", fmri_masked.shape)
    print("iterator.shape", iterator.shape)
    return iterator


if __name__ == '__main__':
    # 所有数据集
    NYU_path = '/ADHD200Data/NYU_dataset/NYU/'
    NYU_filenames = prepare_filenames(NYU_path)
    print("NYU_filenames.num:", len(NYU_filenames))
    print("NYU_filenames", NYU_filenames)
    NYU_data = prepare_data(NYU_filenames)
    print("NYU_data.shape\n", NYU_data.shape)
    np.save('/ADHD200Data/NYU_dataset/NYU/NYU_data.npy', NYU_data)
