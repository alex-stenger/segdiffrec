#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Date   : 2023/6/23 22:26
    @Author : chairc
    @Site   : https://github.com/chairc
"""
import torch
import torch.nn as nn

from model.networks.base import BaseNet
from model.modules.attention import SelfAttention
from model.modules.block import DownBlock, UpBlock
from model.modules.conv import DoubleConv

class UNetLight(BaseNet):
    def __init__(self, in_channel=3, out_channel=3, channel=[3, 32, 64, 128, 256], time_channel=256, num_classes=None, image_size=64, device="cpu", act="relu"):
        super().__init__(in_channel, out_channel, channel, time_channel, num_classes, image_size, device, act)

        self.inc = DoubleConv(in_channels=self.in_channel, out_channels=self.channel[1], act=self.act)
        self.down1 = DownBlock(in_channels=self.channel[1], out_channels=self.channel[2], act=self.act)
        self.down2 = DownBlock(in_channels=self.channel[2], out_channels=self.channel[3], act=self.act)
        self.down3 = DownBlock(in_channels=self.channel[3], out_channels=self.channel[3], act=self.act)

        self.bot1 = DoubleConv(in_channels=self.channel[3], out_channels=self.channel[4], act=self.act)
        self.bot2 = DoubleConv(in_channels=self.channel[4], out_channels=self.channel[4], act=self.act)
        self.bot3 = DoubleConv(in_channels=self.channel[4], out_channels=self.channel[3], act=self.act)

        self.up1 = UpBlock(in_channels=self.channel[4], out_channels=self.channel[2], act=self.act)
        self.up2 = UpBlock(in_channels=self.channel[3], out_channels=self.channel[1], act=self.act)
        self.up3 = UpBlock(in_channels=self.channel[2], out_channels=self.channel[1], act=self.act)

        self.outc = nn.Conv2d(in_channels=self.channel[1], out_channels=self.out_channel, kernel_size=1)

    def forward(self, x, time, y=None):
        time = time.unsqueeze(-1).type(torch.float)
        time = self.pos_encoding(time, self.time_channel)

        if y is not None:
            time += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, time)
        x3 = self.down2(x2, time)
        x4 = self.down3(x3, time)

        bot1_out = self.bot1(x4)
        bot2_out = self.bot2(bot1_out)
        bot3_out = self.bot3(bot2_out)

        up1_out = self.up1(bot3_out, x3, time)
        up2_out = self.up2(up1_out, x2, time)
        up3_out = self.up3(up2_out, x1, time)

        output = self.outc(up3_out)
        return output


class UNetAttention(BaseNet):
    """
    UNet
    """

    def __init__(self, in_channel=6, out_channel=3, channel=[3, 64, 128, 256, 512], time_channel=256, num_classes=None, image_size=64,
                 device="cpu", act="silu"):
        """
        Initialize the UNet network
        :param in_channel: Input channel
        :param out_channel: Output channel
        :param channel: The list of channel
        :param time_channel: Time channel
        :param num_classes: Number of classes
        :param image_size: Adaptive image size
        :param device: Device type
        :param act: Activation function
        """
        super().__init__(in_channel, out_channel, channel, time_channel, num_classes, image_size, device, act)

        # channel: 3 -> 64
        # size: size
        self.inc = DoubleConv(in_channels=self.in_channel, out_channels=self.channel[1], act=self.act)

        # channel: 64 -> 128
        # size: size / 2
        self.down1 = DownBlock(in_channels=self.channel[1], out_channels=self.channel[2], act=self.act)
        # channel: 128
        # size: size / 2
        self.sa1 = SelfAttention(channels=self.channel[2], size=int(self.image_size / 2), act=self.act)
        # channel: 128 -> 256
        # size: size / 4
        self.down2 = DownBlock(in_channels=self.channel[2], out_channels=self.channel[3], act=self.act)
        # channel: 256
        # size: size / 4
        self.sa2 = SelfAttention(channels=self.channel[3], size=int(self.image_size / 4), act=self.act)
        # channel: 256 -> 256
        # size: size / 8
        self.down3 = DownBlock(in_channels=self.channel[3], out_channels=self.channel[3], act=self.act)
        # channel: 256
        # size: size / 8
        self.sa3 = SelfAttention(channels=self.channel[3], size=int(self.image_size / 8), act=self.act)

        # channel: 256 -> 512
        # size: size / 8
        self.bot1 = DoubleConv(in_channels=self.channel[3], out_channels=self.channel[4], act=self.act)
        # channel: 512 -> 512
        # size: size / 8
        self.bot2 = DoubleConv(in_channels=self.channel[4], out_channels=self.channel[4], act=self.act)
        # channel: 512 -> 256
        # size: size / 8
        self.bot3 = DoubleConv(in_channels=self.channel[4], out_channels=self.channel[3], act=self.act)

        # channel: 512 -> 128   in_channels: up1(512) = down3(256) + bot3(256)
        # size: size / 4
        self.up1 = UpBlock(in_channels=self.channel[4], out_channels=self.channel[2], act=self.act)
        # channel: 128
        # size: size / 4
        self.sa4 = SelfAttention(channels=self.channel[2], size=int(self.image_size / 4), act=self.act)
        # channel: 256 -> 64   in_channels: up2(256) = sa4(128) + sa1(128)
        # size: size / 2
        self.up2 = UpBlock(in_channels=self.channel[3], out_channels=self.channel[1], act=self.act)
        # channel: 128
        # size: size / 2
        self.sa5 = SelfAttention(channels=self.channel[1], size=int(self.image_size / 2), act=self.act)
        # channel: 128 -> 64   in_channels: up3(128) = sa5(64) + inc(64)
        # size: size
        self.up3 = UpBlock(in_channels=self.channel[2], out_channels=self.channel[1], act=self.act)
        # channel: 128
        # size: size
        self.sa6 = SelfAttention(channels=self.channel[1], size=int(self.image_size), act=self.act)

        # channel: 64 -> 3
        # size: size
        self.outc = nn.Conv2d(in_channels=self.channel[1], out_channels=self.out_channel, kernel_size=1)

    def forward(self, x, time, y=None):
        """
        Forward
        :param x: Input
        :param time: Time
        :param y: Input label
        :return: output
        """
        time = time.unsqueeze(-1).type(torch.float)
        time = self.pos_encoding(time, self.time_channel)

        if y is not None:
            time += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, time)
        x2_sa = self.sa1(x2)
        x3 = self.down2(x2_sa, time)
        x3_sa = self.sa2(x3)
        x4 = self.down3(x3_sa, time)
        x4_sa = self.sa3(x4)

        bot1_out = self.bot1(x4_sa)
        bot2_out = self.bot2(bot1_out)
        bot3_out = self.bot3(bot2_out)

        up1_out = self.up1(bot3_out, x3_sa, time)
        up1_sa_out = self.sa4(up1_out)
        up2_out = self.up2(up1_sa_out, x2_sa, time)
        up2_sa_out = self.sa5(up2_out)
        up3_out = self.up3(up2_sa_out, x1, time)
        up3_sa_out = self.sa6(up3_out)
        output = self.outc(up3_sa_out)
        return output
    
    def forward_latent(self, x, time, y=None):
        """
        Forward to get the latent space
        :param x: Input
        :param time: Time
        :param y: Input label
        :return: Latent space
        """
        time = time.unsqueeze(-1).type(torch.float)
        time = self.pos_encoding(time, self.time_channel)

        if y is not None:
            time += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, time)
        x2_sa = self.sa1(x2)
        x3 = self.down2(x2_sa, time)
        x3_sa = self.sa2(x3)
        x4 = self.down3(x3_sa, time)
        x4_sa = self.sa3(x4)

        bot1_out = self.bot1(x4_sa)
        bot2_out = self.bot2(bot1_out)
        bot3_out = self.bot3(bot2_out)
        
        # Return the desired latent space
        return bot3_out,x3_sa,x2_sa,x1,time #To give to the YNet
    
    def freeze_weights_from_inc_to_bot3(self):
        # Freeze weights from inc to bot3
        for layer in [self.inc, self.down1, self.sa1, self.down2, self.sa2, self.down3, self.sa3, self.bot1, self.bot2, self.bot3]:
            for param in layer.parameters():
                param.requires_grad = False


class YNet(nn.Module):
    def __init__(self, out_channel=3, channel=[3, 64, 128, 256, 512], image_size=64, act="silu"):
        super(YNet, self).__init__()

        # channel: 512 -> 128   in_channels: up1(512) = down3(256) + bot3(256)
        # size: size / 4
        self.up1 = UpBlock(in_channels=channel[4], out_channels=channel[2], act=act)
        # channel: 128
        # size: size / 4
        self.sa4 = SelfAttention(channels=channel[2], size=int(image_size / 4), act=act)
        # channel: 256 -> 64   in_channels: up2(256) = sa4(128) + sa1(128)
        # size: size / 2
        self.up2 = UpBlock(in_channels=channel[3], out_channels=channel[1], act=act)
        # channel: 128
        # size: size / 2
        self.sa5 = SelfAttention(channels=channel[1], size=int(image_size / 2), act=act)
        # channel: 128 -> 64   in_channels: up3(128) = sa5(64) + inc(64)
        # size: size
        self.up3 = UpBlock(in_channels=channel[2], out_channels=channel[1], act=act)
        # channel: 128
        # size: size
        self.sa6 = SelfAttention(channels=channel[1], size=int(image_size), act=act)

        # channel: 64 -> 3
        # size: size
        self.outc = nn.Conv2d(in_channels=channel[1], out_channels=out_channel, kernel_size=1)

    def forward(self, bot3_out, x3_sa, x2_sa, x1, time):

        up1_out = self.up1(bot3_out, x3_sa, time)
        up1_sa_out = self.sa4(up1_out)
        up2_out = self.up2(up1_sa_out, x2_sa, time)
        up2_sa_out = self.sa5(up2_out)
        up3_out = self.up3(up2_sa_out, x1, time)
        up3_sa_out = self.sa6(up3_out)
        output = self.outc(up3_sa_out)
        return output


if __name__ == "__main__":
    # Unconditional
    net = UNet(device="cpu", image_size=128)
    # Conditional
    # net = UNet(num_classes=10, device="cpu", image_size=128)
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(1, 3, 128, 128)
    t = x.new_tensor([500] * x.shape[0]).long()
    y = x.new_tensor([1] * x.shape[0]).long()
    print(net(x, t).shape)
    # print(net(x, t, y).shape)
