import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import config


def real_conv(in_ch, out_ch, **kwargs):
    conv = {
        1: nn.Conv1d,
        2: nn.Conv2d,
        3: nn.Conv3d
    }[config.spatial_dimentions]
    if 'kernel_size' not in kwargs:
        kwargs['kernel_size'] = config.kernel_size
    return conv(
        in_ch,
        out_ch,
        bias=config.bias,
        **kwargs
    )


def activation(in_channels=None, **kwargs):
    return {
        'ReLU':   nn.ReLU,
        'LeakyReLU': nn.LeakyReLU,
        'ELU':   nn.ELU
    }[config.activation](**kwargs)


def batch_norm(in_channels=None, **kwargs):
    bn = {
        1: nn.BatchNorm1d,
        2: nn.BatchNorm2d,
        3: nn.BatchNorm3d
    }[config.spatial_dimentions]
    return bn(in_channels, **kwargs)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            real_conv(in_ch, out_ch, padding=1),
            batch_norm(out_ch),
            activation(out_ch),
            nn.Dropout(config.dropout_ratio),
            real_conv(out_ch, out_ch, padding=1),
            batch_norm(out_ch),
            activation(out_ch),
            nn.Dropout(config.dropout_ratio)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DownConv(nn.Module):
    def __init__(self, in_ch):
        super(DownConv, self).__init__()
        self.conv = nn.Sequential(
            real_conv(in_ch, in_ch, stride=2, padding=1),
            batch_norm(in_ch),
            activation(in_ch)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = torch.permute(x, (1, 0, 4, 2, 3))[0]
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.down_conv = DownConv(in_ch)
        self.double_conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        down_x = self.down_conv(x)
        x = self.double_conv(down_x)
        return x, down_x


class BottleNeck(nn.Module):
    def __init__(self, in_ch, out_ch, residual_connection=True):
        super(BottleNeck, self).__init__()
        self.residual_connection = residual_connection
        self.down_conv = DownConv(in_ch)
        self.double_conv = nn.Sequential(
            real_conv(in_ch, 2 * in_ch, padding=1),
            batch_norm(2 * in_ch),
            activation(2 * in_ch),
            nn.Dropout(config.dropout_ratio),
            real_conv(2 * in_ch, out_ch, padding=1),
            batch_norm(out_ch),
            activation(out_ch),
            nn.Dropout(config.dropout_ratio)
        )

    def forward(self, x):
        down_x = self.down_conv(x)
        if self.residual_connection:
            x = self.double_conv(down_x) + down_x
        else:
            x = self.double_conv(down_x)

        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv = nn.Sequential(
            real_conv(in_ch * 2, in_ch, padding=1),
            batch_norm(in_ch),
            activation(in_ch),
            nn.Dropout(config.dropout_ratio),
            real_conv(in_ch, out_ch, padding=1),
            batch_norm(out_ch),
            activation(out_ch),
            nn.Dropout(config.dropout_ratio)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = real_conv(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = torch.unsqueeze(torch.permute(x, (0, 2, 3, 1)), 1)
        return x
