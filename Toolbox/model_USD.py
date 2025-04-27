import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== SCFConv 模块 ====================
class SCFConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(SCFConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.groups = groups

        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, (list, tuple)) and len(stride) == 2:
            self.stride = tuple(stride)
        else:
            raise ValueError(f"Stride must be int or tuple of 2 ints, got {stride}")

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        reduction = max(in_channels // 16, 1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduction, 1),
            nn.ReLU(inplace=True)
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(reduction, kernel_size * kernel_size, 1),
            nn.Sigmoid()
        )
        self.channel_att = nn.Sequential(
            nn.Conv2d(reduction, in_channels, 1),
            nn.Sigmoid()
        )
        self.filter_att = nn.Sequential(
            nn.Conv2d(reduction, out_channels, 1),
            nn.Sigmoid()
        )

        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (BCHW), got {x.shape}")

        B, _, _, _ = x.shape
        k = self.kernel_size
        att = self.attention(x)
        spatial_att = self.spatial_att(att).view(B, 1, k, k)
        channel_att = self.channel_att(att).view(B, self.in_channels, 1, 1)
        filter_att = self.filter_att(att).view(B, self.out_channels, 1, 1, 1)

        weight = self.weight.unsqueeze(0)
        weight = weight * spatial_att.unsqueeze(1) * channel_att.unsqueeze(2) * filter_att
        weight = weight.mean(dim=0)
        weight = weight.contiguous().to(x.dtype).to(x.device)

        out = F.conv2d(x, weight, stride=tuple(self.stride), padding=self.padding, groups=self.groups)
        return out

# ==================== GCE 模块 ====================
class GCE(nn.Module):
    def __init__(self, channels):
        super(GCE, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1, groups=channels)
        self.scfconv = SCFConv(channels, channels, 3, stride=1, padding=1, groups=channels)
        self.conv2 = nn.Conv2d(channels, channels, 1, groups=channels)

        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.scfconv(x)
        x = self.conv2(x)
        return x

# ==================== 空间下采样模块 ====================
class SpatialDownsample(nn.Module):
    def __init__(self, C_m, n):
        super(SpatialDownsample, self).__init__()
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"n must be a positive integer, got {n}")
        self.scfconv = SCFConv(C_m, C_m, kernel_size=n, stride=n)
        self.gce = GCE(C_m)

    def forward(self, Y):
        Y_l1 = self.scfconv(Y)
        Y_l1 = self.gce(Y_l1)
        return Y_l1

# ==================== 光谱下采样模块 ====================
class SpectralDownsample(nn.Module):
    def __init__(self, C, C_m):
        super(SpectralDownsample, self).__init__()
        if C <= C_m:
            raise ValueError(f"C ({C}) must be greater than C_m ({C_m})")
        self.convs = nn.ModuleList()
        in_ch = C
        while in_ch > C_m:
            out_ch = max(in_ch // 2, C_m)
            self.convs.append(nn.Conv2d(in_ch, out_ch, 1))
            in_ch = out_ch

        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, X):
        out = X
        for conv in self.convs:
            out = conv(out)
        return out

# ==================== 光谱上采样模块 ====================
class SpectralUpsample(nn.Module):
    def __init__(self, C_m, C, negative_slope=0.1):
        super(SpectralUpsample, self).__init__()
        if C <= C_m:
            raise ValueError(f"C ({C}) must be greater than C_m ({C_m})")

        self.layers = nn.ModuleList()
        in_ch = C_m  # Start with multispectral channels (e.g., 8)
        # Progressively double channels until reaching or exceeding C
        while in_ch < C:
            out_ch = min(in_ch * 2, C)  # Double channels, cap at C
            self.layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=1))
            if out_ch < C:  # No LeakyReLU after the final convolution
                self.layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
            in_ch = out_ch

        # Initialize weights
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, Y):
        out = Y
        for layer in self.layers:
            out = layer(out)
        return out

# ==================== UDS2C2 主模型 ====================
class UDS2C2(nn.Module):
    def __init__(self, C, C_m, n):
        super(UDS2C2, self).__init__()
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"n must be a positive integer, got {n}")
        if C <= C_m:
            raise ValueError(f"C ({C}) must be greater than C_m ({C_m})")

        self.spatial_downsample = SpatialDownsample(C_m, n)
        self.spectral_downsample = SpectralDownsample(C, C_m)
        self.spectral_upsample = SpectralUpsample(C_m, C, n)

    def forward(self, X, Y):
        if X.dim() != 4 or Y.dim() != 4:
            raise ValueError(f"Expected 4D inputs (BCHW), got X shape {X.shape}, Y shape {Y.shape}")

        Y_l1 = self.spatial_downsample(Y)
        Y_l2 = self.spectral_downsample(X)
        Z = self.spectral_upsample(Y_l1)
        return Z, Y_l1, Y_l2