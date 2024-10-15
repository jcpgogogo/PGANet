# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base
import sys
# sys.path.append('/home/LiuHanChao/work/NAFNet-main/basicsr/models/archs')
# from pytorch_wavelets.dwt.transform2d import DWTForward,DWTInverse
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv6 = nn.Conv2d(in_channels=ffn_channel, out_channels=ffn_channel, kernel_size=3, padding=1, stride=1, groups=ffn_channel,
                               bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.alpha1 = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.beta1 = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.gamma1 = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        #self.ctb = nn.Sequential(*self.ctb)

    def forward(self, inp):
        int = torch.squeeze(inp[0:1,:,:,:,:],0)
        x = torch.squeeze(inp[0:1,:,:,:,:],0)
        reaction=torch.squeeze(inp[1:,:,:,:,:],0)
        
        x = self.norm1(x)

        x = self.conv1(x)
        
        x = self.conv2(x)
        #x = self.ctb(x)
        x = self.sg(x)
        
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = int + x * self.beta +self.gamma1*(reaction-int)

        x = self.conv4(self.norm2(y))
        x=self.conv6(x)
        x = self.sg(x)
        
        x = self.conv5(x)

        x = self.dropout2(x)
        out=self.beta1*y + x * self.gamma +self.alpha1*(reaction-int)
        #out=y + x * self.gamma
        return torch.cat((torch.unsqueeze(out,0),torch.unsqueeze(reaction,0)),0)

class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2
        # mid = []
        # self.middle_blk_num=middle_blk_num
        # for i in range(middle_blk_num):
        #     mid.append(NAFBlockmid(chan))
        #self.midblocks = nn.Sequential(*mid)
        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)
        # self.alpha1 = nn.Parameter(torch.Tensor(1).fill_(0.5),requires_grad=True)
        # self.beta1 = nn.Parameter(torch.Tensor(1).fill_(0.5),requires_grad=True)
        # self.c1 = nn.Parameter(torch.Tensor(1).fill_(1),requires_grad=True)
        # self.cc1 = nn.Parameter(torch.Tensor(1).fill_(1),requires_grad=True)
        #self.dwt = DWTForward(J=1, wave='haar', mode='zero').cuda()
        #self.idwt = DWTInverse(wave='haar', mode='zero').cuda()

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        # Yl,Yh = self.dwt(inp)
        # yhreshape=Yh[0].view(Yl.size()[0],-1,Yl.size()[2],Yl.size()[3])
        # wtfeature = torch.cat((Yl,yhreshape),1)
        # x = self.intro(wtfeature)
        x = self.intro(inp)

        encs = []
        k=1
        for encoder, down in zip(self.encoders, self.downs):
            x=torch.cat((torch.unsqueeze(x,0),torch.unsqueeze(x,0)),0)
            x = encoder(x)
            x=torch.squeeze(x[0:1,:,:,:,:],0)
            encs.append(x)
            x = down(x)
        x=torch.cat((torch.unsqueeze(x,0),torch.unsqueeze(x,0)),0)
        x = self.middle_blks(x)
        x=torch.squeeze(x[0:1,:,:,:,:],0)
        # reaction=x
        # for i in range(self.middle_blk_num):
        #     phi = torch.tanh(self.c1) * (i + 1) ** (-torch.sigmoid(self.alpha1))
        #     psi = torch.tanh(self.cc1) * (i + 1) ** (-torch.sigmoid(self.beta1))
        #     x= self.midblocks[i](x, reaction, phi, psi)



        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x=torch.cat((torch.unsqueeze(x,0),torch.unsqueeze(x,0)),0)
            x = decoder(x)
            x=torch.squeeze(x[0:1,:,:,:,:],0)

        x = self.ending(x)
        # x = x + inp

        # IYl = x[:, 0:3, :, :]
        # IYh = x[:, 3:12, :, :]
        #IYl = torch.unsqueeze(IYl, 1)
        #IYh = torch.unsqueeze(IYh, 1)
        # IYhi = []
        # IYhi.append(IYh.contiguous())
        #IYh=torch.reshape(IYh,(IYh.size()[0],3,3,IYh.size()[2],IYh.size()[3]))
        #IYh = torch.unsqueeze(IYh, 1)
        # IYhi = []
        # IYhi.append(IYh.contiguous())
        # x=self.idwt((IYl, IYhi))
        x = x + inp
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class NAFNetLocal(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    img_channel = 3
    width = 32

    enc_blks = [2, 2, 4, 8]
    middle_blk_num = 12
    dec_blks = [2, 2, 2, 2]

    # enc_blks = [1, 1, 1, 28]
    # middle_blk_num = 1
    # dec_blks = [1, 1, 1, 1]
    
    net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)


    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)
