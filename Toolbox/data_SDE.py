import torch
import torch.utils.data as data
import numpy as np
import h5py
import torchvision


class Dataset(data.Dataset):
    def __init__(self, file_path, name):
        super(Dataset, self).__init__()
        dataset = h5py.File(file_path, 'r')

        # ms = np.array(dataset['ms'][name], dtype=np.float32) / 2047.0  #worldview3
        # lms = np.array(dataset['lms'][name], dtype=np.float32) / 2047.0
        # pan = np.array(dataset['pan'][name], dtype=np.float32) / 2047.0

        ms = np.array(dataset['ms'], dtype=np.float32) / 2700.0  # crism23
        lms = np.array(dataset['lms'], dtype=np.float32) / 2700.0
        pan = np.array(dataset['pan'], dtype=np.float32) / 2700.0

        ms = torch.from_numpy(ms).float()
        lms = torch.from_numpy(lms).float()
        pan = torch.from_numpy(pan).float()

        MS_crop = torchvision.transforms.TenCrop(ms.shape[1] / 2)
        self.ms_crops = MS_crop(ms)
        LMS_crop = torchvision.transforms.TenCrop(lms.shape[1] / 2)
        self.lms_crops = LMS_crop(lms)
        PAN_crop = torchvision.transforms.TenCrop(pan.shape[1] / 2)
        self.pan_crops = PAN_crop(pan)

    def __getitem__(self, item):
        return self.ms_crops[item], self.lms_crops[item], self.pan_crops[item]

    def __len__(self):
        return len(self.ms_crops)
