#!/usr/bin/env python

import torch

from deepinpy.utils import utils


class UNetBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels=2,
        n_classes=2,
        depth=5,
        wf=6,
        padding=True,
        batch_norm=False,
        up_mode='upconv',
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNetBlock, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = torch.nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
                
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = torch.nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)
        self.last = torch.nn.Conv2d(prev_channels, n_classes, kernel_size=1)


class UNetConvBlock(torch.nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(torch.nn.Conv2d(in_size, out_size, kernel_size=3, stride = 1, padding= 1))
        block.append(torch.nn.ReLU())
        if batch_norm:
            block.append(torch.nn.BatchNorm2d(out_size))

        block.append(torch.nn.Conv2d(out_size, out_size, kernel_size=3, stride = 1, padding = 1))
        block.append(torch.nn.ReLU())
        # block.append(torch.nn.Dropout2d(p=0.05))
        if batch_norm:
            block.append(torch.nn.BatchNorm2d(out_size))
        self.block = torch.nn.Sequential(*block)
        

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(torch.nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = torch.nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding = 1)
        elif up_mode == 'upsample':
            self.up = torch.nn.Sequential(
                torch.nn.Upsample(mode='bilinear', scale_factor=2),
                torch.nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)
        return out


class UNet(torch.nn.Module):
    def __init__(self, in_channels=2, latent_channels=64, num_blocks=3, kernel_size=7, bias=False, batch_norm=True, dropout=0, topk=None, l1lam=None, l2lam = None):
        super(UNet, self).__init__()

        self.batch_norm = batch_norm
        self.num_blocks = num_blocks
        self.l2lam = torch.nn.Parameter(torch.tensor(l2lam))
        # initialize conv variables
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.dropout = dropout

        self.UNetBlocks = self._build_model()

        # self.l1lam = l1lam
        self.l1lam = 0.05
        if self.l1lam:
            self.threshold = torch.nn.Softshrink(self.l1lam)

        self.topk = topk

    def forward(self, x):
      x = x.permute(0, 3, 2, 1)
      blocks = []
      for i, down in enumerate(self.UNetBlocks[0].down_path):
          x = down(x)
          if i != len(self.UNetBlocks[0].down_path) - 1:
              blocks.append(x)
              x = torch.nn.functional.max_pool2d(x, 2)

      for i, up in enumerate(self.UNetBlocks[0].up_path):
          x = up(x, blocks[-i - 1])
      last = self.UNetBlocks[0].last(x)
      last = last.permute(0, 3, 2, 1)
      return last


    def _build_model(self):
        UNetBlocks = torch.nn.ModuleList()
        UNetBlocks.append(self._add_block())
        return UNetBlocks

    def _add_block(self, in_channels = 2, out_channels = 2):
        return UNetBlock(in_channels=2,
        n_classes=2,
        depth=4,
        wf=5,
        padding=True,
        batch_norm=False,
        up_mode='upsample')
