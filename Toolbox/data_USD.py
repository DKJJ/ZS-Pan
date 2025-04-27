import torch
import torch.utils.data as data
import numpy as np
import h5py
import torchvision
import scipy.io as sio
import os

class FusionDataset_test(data.Dataset):
    def __init__(self, data_dir):

        self.data_dir = data_dir
        # 获取目录下所有 .mat 文件
        self.mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
        if not self.mat_files:
            raise ValueError(f"No .mat files found in {data_dir}")

    def __len__(self):

        return len(self.mat_files)

    def __getitem__(self, idx):

        mat_path = os.path.join(self.data_dir, self.mat_files[idx])

        # 加载 .mat 文件
        mat_data = sio.loadmat(mat_path)

        # 提取字典中的数据
        gt = mat_data['gt']  # 短波红外高分辨率
        lrhs = mat_data['lrhs']  # 短波红外低分辨率
        uphs = mat_data['uphs']  # 短波红外上采样
        hrms = mat_data['hrms']  # 可见光多光谱（8波段）

        # 确保数据是 NumPy 数组并转换为 float32
        gt = np.asarray(gt, dtype=np.float32)
        lrhs = np.asarray(lrhs, dtype=np.float32)
        uphs = np.asarray(uphs, dtype=np.float32)
        hrms = np.asarray(hrms, dtype=np.float32)


        # 转换为 PyTorch 张量
        gt = torch.from_numpy(gt)
        lrhs = torch.from_numpy(lrhs)
        uphs = torch.from_numpy(uphs)
        hrms = torch.from_numpy(hrms)

        # 返回字典
        return gt, hrms, lrhs, uphs

def data_augmentation(label, mode=0):
    """
    Apply spatial data augmentation to HWC format array, ensuring contiguous output.

    Args:
        label (np.ndarray): Input array in HWC format [H, W, C].
        mode (int): Augmentation mode (0-7):
            0: Original
            1: Horizontal flip
            2: Vertical flip
            3: Horizontal + Vertical flip
            4: 90° rotation
            5: 180° rotation
            6: 270° rotation
            7: 90° rotation + Horizontal flip

    Returns:
        np.ndarray: Augmented array in HWC format, contiguous.
    """
    if mode == 0:
        return label
    elif mode == 1:
        return np.ascontiguousarray(np.fliplr(label))  # Horizontal flip
    elif mode == 2:
        return np.ascontiguousarray(np.flipud(label))  # Vertical flip
    elif mode == 3:
        return np.ascontiguousarray(np.flipud(np.fliplr(label)))  # Horizontal + Vertical flip
    elif mode == 4:
        return np.ascontiguousarray(np.rot90(label, k=1))  # 90° rotation
    elif mode == 5:
        return np.ascontiguousarray(np.rot90(label, k=2))  # 180° rotation
    elif mode == 6:
        return np.ascontiguousarray(np.rot90(label, k=3))  # 270° rotation
    elif mode == 7:
        return np.ascontiguousarray(np.fliplr(np.rot90(label, k=1)))  # 90° rotation + Horizontal flip
    else:
        raise ValueError(f"Invalid augmentation mode: {mode}")

class FusionDataset(data.Dataset):
    def __init__(self, data_dir, augment=False, augment_factor=8):
        """
        FusionDataset with preloading and optional data augmentation.

        Args:
            data_dir (str): Path to .mat files directory.
            augment (bool): Whether to apply data augmentation.
            augment_factor (int): Number of augmentation modes (default 8; use 2 for subset).
        """
        self.data_dir = data_dir
        self.augment = augment
        self.factor = augment_factor if augment else 1  # 8 modes or 2 for testing
        self.data = []

        mat_files = [f for f in sorted(os.listdir(data_dir)) if f.endswith('.mat')]
        if not mat_files:
            raise ValueError(f"No .mat files found in {data_dir}")

        print(f"Preloading {len(mat_files)} .mat files from {data_dir}...")
        for mat_file in mat_files:
            mat_path = os.path.join(data_dir, mat_file)
            try:
                mat_data = sio.loadmat(mat_path)
                lrhs = np.asarray(mat_data['lrhs'], dtype=np.float32)  # [128, 32, 32]
                hrms = np.asarray(mat_data['hrms'], dtype=np.float32)  # [8, 128, 128]
                if lrhs.shape != (128, 32, 32) or hrms.shape != (8, 128, 128):
                    print(f"Warning: Unexpected shapes in {mat_file}: "
                          f"lrhs={lrhs.shape}, hrms={hrms.shape}")
                    continue
                # Optional: Channel-wise normalization (uncomment if needed)
                # lrhs = (lrhs - lrhs.mean(axis=(1, 2), keepdims=True)) / (lrhs.std(axis=(1, 2), keepdims=True) + 1e-6)
                # hrms = (hrms - hrms.mean(axis=(1, 2), keepdims=True)) / (hrms.std(axis=(1, 2), keepdims=True) + 1e-6)
                self.data.append((lrhs, hrms))
            except Exception as e:
                print(f"Error loading {mat_file}: {e}")
                continue

        if not self.data:
            raise ValueError(f"No valid .mat files loaded from {data_dir}")
        print(f"Loaded {len(self.data)} samples into memory. Memory usage: ~{len(self.data) * 1}MB")

    def __len__(self):
        return len(self.data) * self.factor

    def __getitem__(self, index):
        """
        Returns the idx-th sample (lrhs, hrms) as PyTorch tensors.

        Args:
            index (int): Index of the sample (including augmentation).

        Returns:
            tuple: (lrhs, hrms) as [1, 128, 32, 32], [1, 8, 128, 128] tensors.
        """
        file_index = index // self.factor
        aug_mode = index % self.factor if self.augment else 0
        lrhs, hrms = self.data[file_index]

        # Convert to HWC and ensure contiguous
        lrhs_hwc = np.ascontiguousarray(lrhs.transpose(1, 2, 0))  # [32, 32, 128]
        hrms_hwc = np.ascontiguousarray(hrms.transpose(1, 2, 0))  # [128, 128, 8]

        # Apply data augmentation
        lrhs_hwc = data_augmentation(lrhs_hwc, mode=aug_mode)
        hrms_hwc = data_augmentation(hrms_hwc, mode=aug_mode)

        # Convert back to CHW and ensure contiguous
        lrhs = np.ascontiguousarray(lrhs_hwc.transpose(2, 0, 1))  # [128, 32, 32]
        hrms = np.ascontiguousarray(hrms_hwc.transpose(2, 0, 1))  # [8, 128, 128]

        # Convert to PyTorch tensors
        lrhs = torch.from_numpy(lrhs).unsqueeze(0)  # [1, 128, 32, 32]
        hrms = torch.from_numpy(hrms).unsqueeze(0)  # [1, 8, 128, 128]

        return lrhs, hrms

