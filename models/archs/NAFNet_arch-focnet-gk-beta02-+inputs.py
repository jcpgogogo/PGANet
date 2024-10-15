import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/home/lhc/work/NAFNet-main')
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base
from basicsr.models.archs.galerkin import simple_attn
from collections import OrderedDict
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock_galerkin(nn.Module):
    def __init__(self, c,i,DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
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

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        blocks=16
        self.gkconv0 = simple_attn(dw_channel//2, blocks)
        self.gkconv1 = simple_attn(dw_channel//2, blocks)
        self.i=i

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        if self.i==0:
            x = self.gkconv0(x, 0)
            x = self.gkconv1(x, 1)
        
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


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

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
    
DEFAULT_N_CONVS_PER_SCALE = [5, 11, 11, 7]
DEFAULT_COMMUNICATION_BETWEEN_SCALES = [
    [(1, 0), (1, 2), (2, 3), (2, 5), (3, 6), (3, 8), (4, 9), (4, 11)],  # 1-2
    [(1, 0), (1, 2), (4, 3), (4, 5), (7, 6), (7, 8), (10, 9), (10, 11)],  # 2-3
    [(1, 0), (1, 1), (4, 2), (4, 3), (7, 4), (7, 5), (10, 6), (10, 7)],  # 3-4
]


def residual_weights_computation(t, beta):
    w = [beta]
    for k in range(t-1, 0, -1):
        w_k = (1 - (1 + beta) / (t - k + 1)) * w[-1]
        w.append(w_k)
    w = w[::-1]
    return w
# class SwitchLayer(Layer):
#     # form what I understand from the code this is what a switch layer looks
#     # like
#     # very difficult to read from this
#     # https://github.com/hsijiaxidian/FOCNet/blob/master/%2Bdagnn/Gate.m
#     # which is used here
#     # https://github.com/hsijiaxidian/FOCNet/blob/master/FracDCNN.m#L360
#     def __init__(self, **kwargs):
#         super(SwitchLayer, self).__init__(**kwargs)
#         self.switch = self.add_weight(
#             'switch_' + str(K.get_uid('switch')),
#             shape=(),
#             initializer=tf.constant_initializer(0.),  # we add a big initializer
#             # to take into account the adjacent scales by default
#             # but not too big because we want to have some gradient flowing
#         )

#     def call(self, inputs):
#         outputs = inputs * tf.sigmoid(self.switch)
#         return outputs
class SwitchLayer(nn.Module):

    # def __init__(self):
    #     super().__init__()
    def forword(self, inputs):
        outputs = inputs * 0.5
        return outputs

class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[],n_scales=4,
            n_filters=128,
            kernel_size=3,
            n_convs_per_scale=DEFAULT_N_CONVS_PER_SCALE,
            communications_between_scales=DEFAULT_COMMUNICATION_BETWEEN_SCALES,beta=0.2):
        super().__init__()

        self.n_scales = n_scales
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.n_convs_per_scale = n_convs_per_scale
        self.communications_between_scales = communications_between_scales
        self.beta = beta
        self.width=width
        chan=width
        
        chan_unpool=chan*2
        self.unpoolings_per_scale = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(chan_unpool*(2**(i_scale)), 2*chan_unpool*(2**(i_scale)), 1, bias=False),
                    nn.PixelShuffle(2)
                )
                for _ in range(len(self.communications_between_scales[i_scale])//2)
            ])
            for i_scale in range(self.n_scales - 1)
        ])
        
        self.poolings_per_scale =nn.ModuleList([
            nn.ModuleList(
            [
                nn.Conv2d(chan*(2**(i_scale)), 2*chan*(2**(i_scale)), 2, 2)
                for _ in range(len(self.communications_between_scales[i_scale])//2)
            ])
            for i_scale in range(self.n_scales - 1)
        ])
        self.concat_per_scale =nn.ModuleList([           
                nn.Conv2d(2*chan*(2**(i_scale)), chan*(2**(i_scale)), kernel_size=1, padding=0, stride=1, groups=1, bias=True)
            for i_scale in range(self.n_scales)
        ])
        # unpooling is not specified in the paper, but in the code
        # you can see a deconv is used
        # https://github.com/hsijiaxidian/FOCNet/blob/master/FracDCNN.m#L415
        self.n_switches_per_scale = []
        self.compute_n_switches_per_scale()
        self.switches_per_scale = nn.ModuleList([
            nn.ModuleList([
                SwitchLayer()
                for _ in range(self.n_switches_per_scale[i_scale])
            ])
            for i_scale in range(self.n_scales)
        ])
        # self.conv_blocks_per_scale = nn.ModuleList([
        #     nn.ModuleList([
        #         NAFBlock_galerkin(chan*(2**(i_scale)),0 if i % 2 else 1) for i in range(self.n_convs_per_scale[i_scale])
        #     ])
        #     for i_scale in range(self.n_scales)
        # ])
        # self.conv_blocks_per_scale = nn.ModuleList([
        #     nn.ModuleList([
        #         NAFBlock_galerkin(chan*(2**(i_scale)),0 if i_scale > 1 else 1) for i in range(self.n_convs_per_scale[i_scale])
        #     ])
        #     for i_scale in range(self.n_scales)
        # ])
        self.conv_blocks_per_scale = nn.ModuleList([
            nn.ModuleList([
                NAFBlock(chan*(2**(i_scale)))  for i in range(self.n_convs_per_scale[i_scale])
            ])
            for i_scale in range(self.n_scales)
        ])
        # self.conv_blocks_per_scale = [
        #     [NAFBlock(chan) for _ in range(n_conv_blocks)]
        #     for n_conv_blocks in self.n_convs_per_scale
        # ]
        self.needs_to_compute = {}
        self.build_needs_to_compute()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        # self.unpoolings_per_scale=nn.ModuleList(*OrderedDict(self.unpoolings_per_scale))
        # self.poolings_per_scale=nn.ModuleList(*self.poolings_per_scale)
        # self.conv_blocks_per_scale=nn.ModuleList(*self.conv_blocks_per_scale)
        # self.switches_per_scale=nn.ModuleList(*self.switches_per_scale)

        # self.padder_size = 2 ** len(self.encoders)
        # self.conv_blocks_per_scale = nn.Sequential(*self.conv_blocks_per_scale)
        # self.switches_per_scale = nn.Sequential(*self.switches_per_scale)
        # self.poolings_per_scale = nn.Sequential(*self.poolings_per_scale)
        # self.unpoolings_per_scale = nn.Sequential(*self.unpoolings_per_scale)
    
    def build_needs_to_compute(self):
        for i_scale, scale_communication in enumerate(self.communications_between_scales):
            down = True
            for i_conv_scale_up, i_conv_scale_down in scale_communication:
                scale_up_node = (i_scale, i_conv_scale_up)
                scale_down_node = (i_scale + 1, i_conv_scale_down)
                if down:
                    self.needs_to_compute[scale_down_node] = scale_up_node
                else:
                    self.needs_to_compute[scale_up_node] = scale_down_node
                down = not down

    def compute_n_switches_per_scale(self):
        for i_scale in range(self.n_scales):
            if i_scale == 0:
                n_switches = len(self.communications_between_scales[0]) // 2
            elif i_scale == self.n_scales - 1:
                n_switches = len(self.communications_between_scales[-1]) // 2
            else:
                n_switches = len(self.communications_between_scales[i_scale - 1]) // 2
                n_switches += len(self.communications_between_scales[i_scale]) // 2
            self.n_switches_per_scale.append(n_switches)

    # def call(self, inputs):
    def forward(self, inputs):
        
        # chan = self.width
        # inp = self.check_image_size(inp)

        # x = self.intro(inp)
        features_per_scale = [[] for _ in range(self.n_scales)]
        features_per_scale[0].append(self.intro(inputs))
        unpoolings_used_per_scale = [0 for _ in range(self.n_scales - 1)]
        poolings_used_per_scale = [0 for _ in range(self.n_scales - 1)]
        switches_used_per_scale = [0 for _ in range(self.n_scales)]
        i_scale = 0
        i_feature = 0
        while i_scale != 0 or i_feature < self.n_convs_per_scale[0]:
            if i_feature >= self.n_convs_per_scale[i_scale]:
                i_scale -= 1
                i_feature = len(features_per_scale[i_scale]) - 1
            node_to_compute = self.needs_to_compute.get(
                (i_scale, i_feature),
                None,
            )
            if node_to_compute is not None:
                i_scale_to_compute, i_feature_to_compute = node_to_compute
                # test if feature is already computed
                n_features_scale_to_compute = len(features_per_scale[i_scale_to_compute])
                if n_features_scale_to_compute <= i_feature_to_compute:
                    # the feature has not been computed, we need to compute it
                    i_scale = i_scale_to_compute
                    i_feature = max(n_features_scale_to_compute - 1, 0)
                    # if there are no features we add it as well
                    continue
                else:
                    # the feature has already been computed we can just use it as is
                    additional_feature = features_per_scale[i_scale_to_compute][i_feature_to_compute]
                    if i_scale_to_compute > i_scale:
                        # the feature has to be unpooled and switched
                        # for now since I don't understand switching, I just do
                        # unpooling, switching will be implemented later on
                        i_unpooling = unpoolings_used_per_scale[i_scale]
                        unpooling = self.unpoolings_per_scale[i_scale][i_unpooling]
                        additional_feature_processed = unpooling(
                            additional_feature,
                        )
                        unpoolings_used_per_scale[i_scale] += 1
                    else:
                        # the feature has to be pooled
                        i_pooling = poolings_used_per_scale[i_scale-1]
                        pooling = self.poolings_per_scale[i_scale-1][i_pooling]
                        additional_feature_processed = pooling(
                            additional_feature,
                        )
                        # unpoolings_used_per_scale[i_scale] += 1
                        # additional_feature_processed = self.pooling(
                        #     additional_feature,
                        # )
                    # i_switch = switches_used_per_scale[i_scale]
                    # switch = self.switches_per_scale[i_scale][i_switch]
                    # additional_feature_processed = switch(additional_feature_processed)
                    # switches_used_per_scale[i_scale] += 1
                    if len(features_per_scale[i_scale]) == 0:
                        # this is the first feature added to the scale
                        features_per_scale[i_scale].append(additional_feature_processed)
                        feature = additional_feature_processed
                    else:
                        feature = torch.concat([
                            features_per_scale[i_scale][i_feature],
                            additional_feature_processed,
                        ], axis=-3)
                        feature=self.concat_per_scale[i_scale](feature)

            else:
                feature = features_per_scale[i_scale][-1]
            conv_block=self.conv_blocks_per_scale[i_scale][i_feature]
            B, C, H, W = feature.shape
            new_feature = conv_block(feature)
            weights = residual_weights_computation(
                i_feature,
                beta=self.beta,
            )
            for weight, res_feature in zip(weights, features_per_scale[i_scale]):
                new_feature = new_feature + weight * res_feature
            features_per_scale[i_scale].append(new_feature)
            i_feature += 1
        outputs = self.ending(features_per_scale[0][self.n_convs_per_scale[0]])
        # this could be -1 instead of self.n_convs_per_scale[0], but it's an
        # extra sanity check that everything is going alright
        outputs=outputs+inputs
        return outputs
    

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

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]
    
    net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)


    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)
