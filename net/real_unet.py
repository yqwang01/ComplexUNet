import torch.nn as nn
import net.real_blocks as unet_real
from configs import config


class RealUNet(nn.Module):
    def __init__(self,):
        super(RealUNet, self).__init__()
        self.inc = unet_real.InConv(2, 128)
        self.down1 = unet_real.Down(128, 256)
        self.down2 = unet_real.Down(256, 512)
        self.down3 = unet_real.Down(512, 1024)
        self.bottleneck = unet_real.BottleNeck(1024, 1024, False)
        self.up1 = unet_real.Up(1024, 512)
        self.up2 = unet_real.Up(512, 256)
        self.up3 = unet_real.Up(256, 128)
        self.up4 = unet_real.Up(128, 128)
        self.ouc = unet_real.OutConv(128, 2)

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