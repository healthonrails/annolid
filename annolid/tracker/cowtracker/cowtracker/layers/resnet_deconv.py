# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""ResNet18-style encoder-decoder for image features."""

import torch.nn as nn


class resconv(nn.Module):
    """Residual convolution block."""

    def __init__(self, inp, oup, k=3, s=1):
        super(resconv, self).__init__()
        self.conv = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(inp, oup, kernel_size=k, stride=s, padding=k // 2, bias=True),
            nn.GELU(),
            nn.Conv2d(oup, oup, kernel_size=3, stride=1, padding=1, bias=True),
        )
        if inp != oup or s != 1:
            self.skip_conv = nn.Conv2d(
                inp, oup, kernel_size=1, stride=s, padding=0, bias=True
            )
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.skip_conv(x)


class ResNet18Deconv(nn.Module):
    """ResNet18-style encoder-decoder for image features."""

    def __init__(self, inp, oup):
        super(ResNet18Deconv, self).__init__()
        self.ds1 = resconv(inp, 64, k=7, s=2)
        self.conv1 = resconv(64, 64, k=3, s=1)
        self.conv2 = resconv(64, 128, k=3, s=2)
        self.conv3 = resconv(128, 256, k=3, s=2)
        self.conv4 = resconv(256, 512, k=3, s=2)
        self.up_4 = nn.ConvTranspose2d(
            512, 256, kernel_size=2, stride=2, padding=0, bias=True
        )
        self.proj_3 = resconv(256, 256, k=3, s=1)
        self.up_3 = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2, padding=0, bias=True
        )
        self.proj_2 = resconv(128, 128, k=3, s=1)
        self.up_2 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2, padding=0, bias=True
        )
        self.proj_1 = resconv(64, oup, k=3, s=1)

    def forward(self, x):
        out_1 = self.ds1(x)
        out_1 = self.conv1(out_1)
        out_2 = self.conv2(out_1)
        out_3 = self.conv3(out_2)
        out_4 = self.conv4(out_3)
        out_3 = self.proj_3(out_3 + self.up_4(out_4))
        out_2 = self.proj_2(out_2 + self.up_3(out_3))
        out_1 = self.proj_1(out_1 + self.up_2(out_2))
        return [out_1, out_2, out_3, out_4]
