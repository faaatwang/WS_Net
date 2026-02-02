from model.network_swinir import SwinIR
from model.DWT import DWT, IWT

import torch
from torch import nn
from losses.myloss import EdgeConnectLoss

class WS_Net(nn.Module):
    def __init__(self, height=224, width=224, window_size=8, num=[6, 6, 6, 6]):
        super(WS_Net, self).__init__()
        self.height = height
        self.width = width
        self.window_size = window_size
        self.num = num
        self.sw1 = nn.Sequential(
                                 nn.GELU(),
                                 SwinIR(upscale=2, img_size=(self.height, self.width),
                                        window_size=self.window_size, img_range=1., depths=self.num,
                                        embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2,
                                        upsampler='nearest+conv'),

                                 )
        self.sw2 = nn.Sequential(
                                 nn.GELU(),
                                 SwinIR(upscale=2, img_size=(self.height, self.width),
                                        window_size=self.window_size, img_range=1., depths=self.num,
                                        embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2,
                                        upsampler='nearest+conv'),

                                 )
        self.sw3 = nn.Sequential(
                                 nn.GELU(),
                                 SwinIR(upscale=2, img_size=(self.height, self.width),
                                        window_size=self.window_size, img_range=1., depths=self.num,
                                        embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2,
                                        upsampler='nearest+conv'),

                                 )
        self.conv_1 = nn.Conv2d(3, 3, 4, 2, 1)
        self.conv_2 = nn.ConvTranspose2d(3, 3, 4, 2, 1)
        self.dwt = DWT()
        self.iwt = IWT()
        self.e_loss = EdgeConnectLoss(gan_loss='nsgan', g_gradient_loss_weight=5.0, disc_start=50000,
                                      content_start=50000, gradient_start=50000, style_start=50000, )

    def forward_features(self, x):
        x = self.conv_1(x)
        x = self.dwt(x)
        x_hh, x_hl, x_lh, x_ll = torch.chunk(x, 4, dim=1)
        x_hh = self.sw1(x_hh)
        x_hl = self.sw2(x_hl)
        x_lh = self.sw3(x_lh)
        x_ll = self.conv_2(x_ll)
        x = torch.cat((x_hh, x_hl, x_lh, x_ll), 1)
        x = self.iwt(x)

        return x

    def forward(self, x_i, x_n, step):
        if self.training:
            x = self.forward_features(x_n)
            e_loss = self.e_loss(image=x_i, reconstruction=x, step=step, name='generator')['loss']
            return x, e_loss
        else:
            return self.forward_features(x_n)
