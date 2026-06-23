# Dual-branch lightweight super-resolution network（DB_LSRNet）

import torch
import torch.nn as nn
import numbers
import math
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
from torch.nn import init as init
from torch.nn import Softmax, Parameter
from torch.nn.modules.batchnorm import _BatchNorm


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


def norm(nc, norm_type):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


class en_TransformerBlock1(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=False, downscale=(5, 5), BatchNorm_type="batch",
                 LayerNorm_type="WithBias", num_block=5):
        super(en_TransformerBlock1, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias),
                                   nn.LeakyReLU(0.2), )
        self.norm1 = norm(dim, BatchNorm_type)
        self.attn = Attention(dim)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        # self.scdm = nn.Sequential(make_layer(ResidualBlockShift, num_block, num_feat=dim),
        #                           DownShiftMLP(dim, scale=downscale),
        #                           )
        # self.down_conv1 = DownShiftMLP(dim, scale=downscale)
        # self.body = make_layer(ResidualBlockShift, num_block, num_feat=dim)

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        # x = self.scdm(x)
        return x


class de_TransformerBlock1(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=False, upscale=(5, 5), BatchNorm_type="batch",
                 LayerNorm_type="WithBias", num_block=5):
        super(de_TransformerBlock1, self).__init__()

        self.norm1 = norm(dim, BatchNorm_type)
        self.attn = Attention(dim)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        # self.scum = nn.Sequential(make_layer(ResidualBlockShift, num_block, num_feat=dim),
        #                           UpShiftMLP(dim, scale=upscale), )
        self.tconv1 = nn.Sequential(nn.ConvTranspose2d(dim, dim, kernel_size=3, stride=1, padding=1, output_padding=0),
                                    nn.LeakyReLU(0.2), )
        # self.upconv1 = UpShiftMLP(dim, scale=upscale)
        # self.body = make_layer(ResidualBlockShift, num_block, num_feat=dim)

    def forward(self, x):
        # x = self.scum(x)
        x = self.tconv1(x)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation,
                     groups=groups)


def activation(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class FourierUnit(nn.Module):
    def __init__(self, dim=8):
        super(FourierUnit, self).__init__()
        self.fpre = nn.Conv2d(dim, dim, 1, 1, 0)
        self.process1 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(dim, dim, 1, 1, 0))
        self.process2 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(dim, dim, 1, 1, 0))
        self.process3 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0),
            # nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(self.fpre(x), norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        pha = self.process2(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        x_out = self.process3(x_out)
        return x_out + x


class FMAMod(nn.Module):
    def __init__(self, dim, num_heads):
        super(FMAMod, self).__init__()
        layer_scale_init_value = 1e-6
        self.num_heads = num_heads
        self.norm = LayerNorm(dim, LayerNorm_type="WithBias")
        self.a = FourierUnit(dim)
        self.v = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(num_heads), requires_grad=True)
        self.CPE = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        shortcut = x
        pos_embed = self.CPE(x)
        x = self.norm(x)
        a = self.a(x)
        v = self.v(x)
        a = rearrange(a, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        a_all = torch.split(a, math.ceil(N // 4), dim=-1)
        v_all = torch.split(v, math.ceil(N // 4), dim=-1)
        attns = []
        for a, v in zip(a_all, v_all):
            attn = a * v
            attn = self.layer_scale.unsqueeze(-1).unsqueeze(-1) * attn
            attns.append(attn)
        x = torch.cat(attns, dim=-1)
        x = F.softmax(x, dim=-1)
        x = rearrange(x, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        x = x + pos_embed
        x = self.proj(x)
        out = x + shortcut

        return out


class Attention(nn.Module):  # 整合多尺度特征，聚焦关键信息（如目标区域），抑制噪声干扰
    def __init__(self, in_channels, distillation_rate=0.25):
        super(Attention, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = nn.Sequential(conv_layer(in_channels, in_channels, 1),
                                activation('lrelu', neg_slope=0.2),
                                )
        self.FMA1 = FMAMod(dim=in_channels, num_heads=8)
        self.c2 = nn.Sequential(conv_layer(in_channels, in_channels, 1),
                                activation('lrelu', neg_slope=0.2),
                                )
        self.FMA2 = FMAMod(dim=in_channels, num_heads=8)
        self.c3 = nn.Sequential(conv_layer(in_channels, in_channels, 1),
                                activation('lrelu', neg_slope=0.2),
                                )
        self.FMA3 = FMAMod(dim=in_channels, num_heads=8)

        self.c4 = conv_layer(in_channels * 4, in_channels, 1)
        self.cca = CCALayer(in_channels * 4)

    def forward(self, input):
        conv_c1 = self.c1(input)
        fma_c1 = self.FMA1(input)
        # distilled_c1, remaining_c1 = torch.split(fma_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        conv_c2 = self.c2(fma_c1)
        fma_c2 = self.FMA2(input + fma_c1)
        # distilled_c2, remaining_c2 = torch.split(fma_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        conv_c3 = self.c3(fma_c2)
        fma_c3 = self.FMA2(input + fma_c2)
        out = torch.cat([conv_c1, conv_c2, conv_c3, input + fma_c3], dim=1)
        out_fused = self.c4(self.cca(out)) + input  # 将 3 个分支的输出与残差输入拼接，通过 1x1 卷积压缩维度，最终与原始输入残差连接。
        return out_fused


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class en_TransformerBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=False, downscale=(5, 5), BatchNorm_type="batch",
                 LayerNorm_type="WithBias", num_block=5):
        super(en_TransformerBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias),
                                   nn.LeakyReLU(0.2), )
        self.norm1 = norm(dim, BatchNorm_type)
        self.attn = Attention(dim)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.scdm = nn.Sequential(
            make_layer(ResidualBlockShift, num_block, num_feat=dim),
            DownShiftMLP(dim),
        )

        # self.down_conv1 = DownShiftMLP(dim, scale=downscale)
        # self.body = make_layer(ResidualBlockShift, num_block, num_feat=dim)

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        x = self.scdm(x)
        return x


class de_TransformerBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=False, upscale=(5, 5), BatchNorm_type="batch",
                 LayerNorm_type="WithBias", num_block=5):
        super(de_TransformerBlock, self).__init__()

        self.norm1 = norm(dim, BatchNorm_type)
        self.attn = Attention(dim)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.scum = nn.Sequential(
            make_layer(ResidualBlockShift, num_block, num_feat=dim),
            UpShiftMLP(dim),
        )

        self.tconv1 = nn.Sequential(nn.ConvTranspose2d(dim, dim, kernel_size=3, stride=1, padding=1, output_padding=0),
                                    nn.LeakyReLU(0.2), )
        # self.upconv1 = UpShiftMLP(dim, scale=upscale)
        # self.body = make_layer(ResidualBlockShift, num_block, num_feat=dim)

    def forward(self, x):
        x = self.scum(x)
        x = self.tconv1(x)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Shift8(nn.Module):
    def __init__(self, groups=4, stride=1, mode='constant') -> None:
        super(Shift8, self).__init__()
        self.g = groups
        self.mode = mode
        self.stride = stride

    def forward(self, x):
        b, c, h, w = x.shape
        out = torch.zeros_like(x)

        pad_x = F.pad(x, pad=[self.stride for _ in range(4)], mode=self.mode)
        assert c == self.g * 8

        cx, cy = self.stride, self.stride
        stride = self.stride
        out[:, 0 * self.g:1 * self.g, :, :] = pad_x[:, 0 * self.g:1 * self.g, cx - stride:cx - stride + h, cy:cy + w]
        out[:, 1 * self.g:2 * self.g, :, :] = pad_x[:, 1 * self.g:2 * self.g, cx + stride:cx + stride + h, cy:cy + w]
        out[:, 2 * self.g:3 * self.g, :, :] = pad_x[:, 2 * self.g:3 * self.g, cx:cx + h, cy - stride:cy - stride + w]
        out[:, 3 * self.g:4 * self.g, :, :] = pad_x[:, 3 * self.g:4 * self.g, cx:cx + h, cy + stride:cy + stride + w]

        out[:, 4 * self.g:5 * self.g, :, :] = pad_x[:, 4 * self.g:5 * self.g, cx + stride:cx + stride + h,
                                              cy + stride:cy + stride + w]
        out[:, 5 * self.g:6 * self.g, :, :] = pad_x[:, 5 * self.g:6 * self.g, cx + stride:cx + stride + h,
                                              cy - stride:cy - stride + w]
        out[:, 6 * self.g:7 * self.g, :, :] = pad_x[:, 6 * self.g:7 * self.g, cx - stride:cx - stride + h,
                                              cy + stride:cy + stride + w]
        out[:, 7 * self.g:8 * self.g, :, :] = pad_x[:, 7 * self.g:8 * self.g, cx - stride:cx - stride + h,
                                              cy - stride:cy - stride + w]

        # out[:, 8*self.g:, :, :] = pad_x[:, 8*self.g:, cx:cx+h, cy:cy+w]
        return out


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


class ResidualBlockShift(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-Shift-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockShift, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, kernel_size=1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, kernel_size=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        # self.relu = nn.ReLU(inplace=True)
        self.shift = Shift8(groups=num_feat // 8, stride=1)

        # if not pytorch_init:
        #     default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.shift(self.conv1(x))))
        return identity + out * self.res_scale

def align_like(src, ref):
    if src.shape[-2:] != ref.shape[-2:]:
        src = F.interpolate(
            src,
            size=ref.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
    return src


class DownShiftMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, 3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),
            Shift8(groups=dim // 8),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x):
        return self.body(x)


class UpShiftMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 3, stride=2, padding=1, output_padding=0, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),
            Shift8(groups=dim // 8),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x):
        return self.body(x)

class en_Backbone(nn.Module):
    def __init__(self, dim=64, num_block=5):
        super().__init__()
        self.body = make_layer(ResidualBlockShift, num_block, num_feat=dim)
        self.down = DownShiftMLP(dim)

    def forward(self, x):
        feat = self.body(x)
        down = self.down(feat)
        return feat, down


class de_Backbone(nn.Module):
    def __init__(self, dim=64, num_block=5):
        super().__init__()
        self.body = make_layer(ResidualBlockShift, num_block, num_feat=dim)
        self.up = UpShiftMLP(dim)

    def forward(self, x):
        x = self.body(x)
        return self.up(x)

class de_Backbone1(nn.Module):
    def __init__(self, in_channels=64, dim=64, num_block=5, upscale=(5, 5)):
        super(de_Backbone1, self).__init__()
        # self.upconv1 = UpShiftMLP(in_channels, scale=upscale)
        # self.body = make_layer(ResidualBlockShift, num_block, num_feat=in_channels)
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
        )
        self.scum = nn.Sequential(make_layer(ResidualBlockShift, num_block, num_feat=dim),
                                  UpShiftMLP(dim, scale=upscale), )

    def forward(self, x):
        x = self.conv2(x)
        out = self.scum(x)
        return out


class SAttention(nn.Module):
    def __init__(self, hidden_features):
        super(SAttention, self).__init__()

        # Define the convolutional layers for K, Q, and V
        self.conv_k1 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1))
        self.conv_q1 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1))
        self.conv_v1 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1))
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, f):
        # Compute K
        k = self.conv_k1(x)
        # Compute Q
        q = self.conv_q1(f)
        # Compute V
        v = self.conv_v1(x)

        # Compute the attention scores S (softmax of K * Q^T)
        batch_size, C, height, width = k.size()
        k = k.view(batch_size, C, -1).permute(0, 2, 1)  # (batch_size, 8, height*width)
        q = q.view(batch_size, C, -1)  # (batch_size, 8, height*width)

        # Compute attention scores
        scores = torch.bmm(q, k)  # (batch_size, height*width, height*width)
        scores_new = torch.max(scores, -1, keepdim=True)[0].expand_as(scores) - scores

        # Apply softmax to get the attention map S
        attention_map = self.softmax(scores_new)  # (batch_size, height*width, height*width)

        # Reshape v to (batch_size, 8, height, width)
        v = v.view(batch_size, C, -1)

        # Perform the matrix multiplication S * V
        out = torch.bmm(attention_map, v)
        out = out.view(batch_size, C, height, width)
        out = self.gamma * out + x
        return self.sigmoid(out)

class DB_LSRNet(nn.Module):
    def __init__(self, in_channels=3, dim=64):
        super().__init__()

        self.fist_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, dim, 1),
            nn.LeakyReLU(0.2)
        )

        self.fist_conv2 = nn.Sequential(
            nn.Conv2d(in_channels, dim, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

        self.fist_conv3 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, in_channels, 3, padding=1),
        )

        # Backbone
        self.en_B1 = en_Backbone(dim)
        self.en_B2 = en_Backbone(dim)
        self.de_B1 = de_Backbone(dim)
        self.de_B2 = de_Backbone(dim)

        # Transformer
        self.en_T1 = en_TransformerBlock(dim)
        self.en_T2 = en_TransformerBlock(dim)
        self.en_T3 = en_TransformerBlock1(dim)

        self.de_T0 = de_TransformerBlock1(dim)
        self.de_T1 = de_TransformerBlock(dim)
        self.de_T2 = de_TransformerBlock(dim)

        self.s_att0 = SAttention(dim)
        self.s_att1 = SAttention(dim)
        self.s_att2 = SAttention(dim)
        self.s_att3 = SAttention(dim)

    def forward(self, x, t):
        # ---- t branch (18x18) ----  风格分支
        t = self.fist_conv1(t)
        Bs1, Bt1 = self.en_B1(t)   #Bs1用于注意力连接  Bt1用于下一层输入
        Bs2, Bt2 = self.en_B2(Bt1)

        Dt1 = align_like(self.de_B1(Bt2), Bt1) + Bt1
        Dt2 = align_like(self.de_B2(Dt1), t) + t

        # ---- x branch (36x36) ----
        x = self.fist_conv2(x)

        T1 = self.s_att0(self.en_T1(x), Bs1)
        T2 = self.s_att1(self.en_T2(T1), Bs2)

        T3 = self.en_T3(T2)
        D0 = align_like(self.de_T0(T3), T2) + T2
        D0 = self.s_att2(D0, Dt1)

        tmp = align_like(self.de_T1(D0), T1)
        # D1 = self.s_att3(tmp + T1, Dt2)
        D1 = tmp + T1 + Dt2
        D2 = align_like(self.de_T2(D1), x) + x

        return self.fist_conv3(D2)


if __name__ == "__main__":
    x = torch.randn(1, 3, 36, 36)
    t = torch.randn(1, 3, 18, 18)

    model = DB_LSRNet(in_channels=3, dim=32)
    y = model(x, t)

    print("Output shape:", y.shape)

