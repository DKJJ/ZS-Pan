import torch
import torch.utils.data as data
import scipy.io as sio
import numpy as np
import h5py
import torchvision


# file_path = '../dataset/crism_23.h5'
file_path = '../dataset/train_wv3.h5'
mat_path = '../dataset/crism_23.mat'
name = 19

dataset = h5py.File(file_path, 'r')

mat_data = sio.loadmat(mat_path)

with h5py.File('../dataset/crism_23.h5', 'w') as h5_file:
    # 遍历 .mat 文件中的每个变量
    for key, value in mat_data.items():
        # 忽略文件中的元数据，如 __header__, __version__, __globals__ 等
        if key.startswith('__'):
            continue
        if key == 'pan':
            save_key = 'pan'
        elif key == 'ref':
            save_key = 'lms'
        else:
            save_key = 'ms'

        # 将每个变量写入 HDF5 文件
        h5_file.create_dataset(save_key, data=value)


# ms = np.array(dataset['ms'][name], dtype=np.float32) / 2047.0  #worldview3
# lms = np.array(dataset['lms'][name], dtype=np.float32) / 2047.0
# pan = np.array(dataset['pan'][name], dtype=np.float32) / 2047.0

ms = np.array(dataset['ms'][name], dtype=np.float32) / 2700.0  #crism23
lms = np.array(dataset['lms'][name], dtype=np.float32) / 2700.0
pan = np.array(dataset['pan'][name], dtype=np.float32) / 2700.0

ms = torch.from_numpy(ms).float()
lms = torch.from_numpy(lms).float()
pan = torch.from_numpy(pan).float()

# MS_crop = torchvision.transforms.TenCrop(ms.shape[1] / 2)
# self.ms_crops = MS_crop(ms)
ms_crops = ms

# LMS_crop = torchvision.transforms.TenCrop(ms.shape[1] / 2)
# self.lms_crops = MS_crop(ms)
lms_crops = lms

# PAN_crop = torchvision.transforms.TenCrop(pan.shape[1] / 2)
# self.pan_crops = PAN_crop(pan)
pan_crops = pan

