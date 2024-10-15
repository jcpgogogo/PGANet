import torch
import torch.nn as nn

# from models import register

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.weight * out + self.bias
        return out

# @register('galerkin')
class simple_attn(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv2d(midc, 3*midc, 1)
        self.o_proj1 = nn.Conv2d(midc, midc, 1)
        self.o_proj2 = nn.Conv2d(midc, midc, 1)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act = nn.GELU()
        dim=midc
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )
    
    def forward(self, x, name='0'):
        B, C, H, W = x.shape
        bias = x
        # qkv = self.qkv_proj(x)
        # qkv = qkv.permute(0, 2, 3, 1)
        # qkv = qkv.reshape(B, H*W, self.heads, 3*self.headc)
        qkv = self.qkv_proj(x).permute(0, 2, 3, 1)
        # _,_,v_conv=qkv.chunk(3, dim=-1)
        # v_conv=v_conv.permute(0, 3, 1, 2)
        qkv = qkv.reshape(B, H*W, self.heads, 3*self.headc).permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        v_conv=q.permute(0, 1, 3, 2).reshape(B, C, H, W)
        k = self.kln(k)
        v = self.vln(v)
        

        
        v = torch.matmul(k.transpose(-2,-1), v) / (H*W)
        v = torch.matmul(q, v)
        v = v.permute(0, 2, 1, 3).reshape(B, H, W, C)

        ret = v.permute(0, 3, 1, 2) + bias

        # convolution output
        conv_x = self.dwconv(v_conv)

        # Adaptive Interaction Module (AIM)
        # C-Map (before sigmoid)
        # channel_map = self.channel_interaction(conv_x).permute(0, 2, 3, 1).contiguous().view(B, 1, C)
        channel_map = self.channel_interaction(conv_x)
        channel_map = channel_map.permute(0, 2, 3, 1).contiguous().view(B, 1, C)
        # S-Map (before sigmoid)
        attention_reshape = ret
        spatial_map = self.spatial_interaction(attention_reshape)

        # C-I
        attened_x = ret.reshape(B, H*W, C) * torch.sigmoid(channel_map)
        # S-I
        conv_x = torch.sigmoid(spatial_map) * conv_x
        conv_x = conv_x.permute(0, 2, 3, 1)

        ret = attened_x.reshape(B, H, W, C).permute(0, 3, 1, 2) + conv_x.permute(0, 3, 1, 2)

        bias = self.o_proj2(self.act(self.o_proj1(ret))) + bias
        
        return bias
    