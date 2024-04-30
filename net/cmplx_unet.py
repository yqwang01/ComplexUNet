import torch.nn as nn
import net.cmplx_blocks as unet_cmplx
from configs import config


class CUNet(nn.Module):
    def __init__(self,):
        super(CUNet, self).__init__()
        self.inc = unet_cmplx.InConv(1, 64)
        self.down1 = unet_cmplx.Down(64, 128)
        self.down2 = unet_cmplx.Down(128, 256)
        self.down3 = unet_cmplx.Down(256, 512)
        self.bottleneck = unet_cmplx.BottleNeck(512, 512, False)
        self.up1 = unet_cmplx.Up(512, 256)
        self.up2 = unet_cmplx.Up(256, 128)
        self.up3 = unet_cmplx.Up(128, 64)
        self.up4 = unet_cmplx.Up(64, 64)
        self.ouc = unet_cmplx.OutConv(64, 1)

    def forward(self, x):
        # x0 = x
        x1 = self.inc(x)
        x2, _ = self.down1(x1)
        x3, _ = self.down2(x2)
        x4, _ = self.down3(x3)
        x5 = self.bottleneck(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # x = x + x0 if config.unet_global_residual_conn else x
        x = self.ouc(x)

        return x
