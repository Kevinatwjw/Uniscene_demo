""" adopted from: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/diffusionmodules/model.py """
# pytorch_diffusion + derived encoder decoder
import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmengine.model import BaseModule
from mmengine.registry import MODELS

from .attention import *
from .vae_utils import shift_dim

# from vae_utils import shift_dim , view_range
# from attention import *


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    if in_channels <= 2:
        num_groups = 2
    elif in_channels <= 32:
        num_groups = in_channels // 4
    else:
        num_groups = 32
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x, shape):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        diffY = shape[0] - x.size()[2]
        diffX = shape[1] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 2, 1)

    def forward(self, x):
        if self.with_conv:
            # pad = (0, 1, 0, 1, 0, 1)
            # x = torch.nn.functional.pad(x, pad, mode='constant', value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb=None):
        h = x
        # h = nonlinearity(h)
        h = self.norm1(h)
        h = self.conv1(h)
        h = F.relu(h)
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        # h = nonlinearity(h)
        h = self.norm2(h)
        h = self.conv2(h)
        h = self.dropout(h)
        h = F.relu(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class AttnBlock3D(nn.Module):
    def __init__(self, in_channels, t_shape):
        super().__init__()
        self.in_channels = in_channels
        self.t_shape = t_shape
        self.norm = nn.BatchNorm3d(in_channels)
        self.q = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        h_ = x
        h_ = rearrange(h_, "(B F) C H W -> B C F H W", F=self.t_shape)
        # print(h_.shape)
        b, c, f, h, w = h_.shape
        # x: shape (B*F,C,H,W)

        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # # compute attention
        # b, c, f, h, w = q.shape
        q = rearrange(q, "B C F H W -> (B C) F (H W)")
        # q = q.reshape(b, f,c*h*w)
        q = q.permute(0, 2, 1)  # bc,hw,f
        # k = k.reshape(b, f, c*h*w) # bc,f,hw
        k = rearrange(k, "B C F H W -> (B C) F (H W)")
        w_ = torch.bmm(q, k)  # bc,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(f) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # # attend to values
        # v = v.reshape(b,f,c*h*w) # b,f, chw
        v = rearrange(v, "B C F H W -> (B C) F (H W)")
        w_ = w_.permute(0, 2, 1)  # bc,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # bc,f,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]

        h_ = h_.reshape(b, c, f, h, w)
        # h_ = h_.permute(0,2,1,3,4) # b c f h w
        h_ = self.proj_out(h_)

        h_ = rearrange(h_, "B C F H W -> (B F) C H W")

        return x + h_


@MODELS.register_module()
class Encoder2D_new(BaseModule):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    print("[*] Enc has Attn at i_level, i_block: %d, %d" % (i_level, i_block))
                    # attn.append(AttnBlock(block_in))
                    attn.append(AxialBlock_wh(block_in, 2))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        self.mid.attn_1 = AxialBlock_wh(block_in, 2)  # AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, 2 * z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        # x: bs, F, H, W, D
        shapes = []
        temb = None

        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):

            for i_block in range(self.num_res_blocks):
                # h = self.down[i_level].block[i_block](hs[-1], temb)
                h = self.down[i_level].block[i_block](h, temb)

                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                # hs.append(h)
            if i_level != self.num_resolutions - 1:
                shapes.append(h.shape[-2:])
                # hs.append(self.down[i_level].downsample(hs[-1]))
                h = self.down[i_level].downsample(h)

        # middle
        # h = hs[-1]
        #
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        # h = nonlinearity(h)
        h = F.relu(h)
        h = self.conv_out(h)
        return h, shapes


@MODELS.register_module()
class VAERes2D_DwT(BaseModule):
    def __init__(self, encoder_cfg, decoder_cfg, num_classes=18, expansion=8, vqvae_cfg=None, init_cfg=None):
        super().__init__(init_cfg)

        self.expansion = expansion
        self.num_cls = num_classes

        self.encoder = MODELS.build(encoder_cfg)
        self.decoder = MODELS.build(decoder_cfg)
        self.class_embeds = nn.Embedding(num_classes, expansion)

        if vqvae_cfg:
            self.vqvae = MODELS.build(vqvae_cfg)
        self.use_vq = vqvae_cfg is not None

    def sample_z(self, z):
        dim = z.shape[1] // 2
        mu = z[:, :dim]
        sigma = torch.exp(z[:, dim:] / 2)
        eps = torch.randn_like(mu)
        return mu + sigma * eps, mu, sigma

    def forward_encoder(self, x):
        # x: bs, F, H, W, D
        bs, F, H, W, D = x.shape
        x = self.class_embeds(x)  # bs, F, H, W, D, c
        x = x.reshape(bs * F, H, W, D * self.expansion).permute(0, 3, 1, 2)

        z, shapes = self.encoder(x)  # bs*F C' H' W'

        z = rearrange(z, "(B F) C H W -> B C F H W", F=F)
        return z, shapes

    def forward_decoder(self, z, input_shape):
        logits = self.decoder(z)  # b c f h w

        bs, F, H, W, D = input_shape
        logits = logits.permute(0, 2, 3, 4, 1).reshape(-1, D, self.expansion)
        template = self.class_embeds.weight.T.unsqueeze(0)  # 1, expansion, cls
        similarity = torch.matmul(logits, template)  # -1, D, cls
        # pred = similarity.argmax(dim=-1) # -1, D
        # pred = pred.reshape(bs, F, H, W, D)
        return similarity.reshape(bs, F, H, W, D, self.num_cls)

    def forward(self, x, **kwargs):
        # xs = self.forward_encoder(x)
        # logits = self.forward_decoder(xs)
        # return logits, xs[-1]

        output_dict = {}
        z, shapes = self.forward_encoder(x)
        if self.use_vq:
            z_sampled, loss, info = self.vqvae(z, is_voxel=True)
            output_dict.update({"embed_loss": loss})
        else:
            z_sampled, z_mu, z_sigma = self.sample_z(z)
            kl_loss = -0.5 * torch.mean(1 + z_sigma - torch.exp(z_sigma) - z_mu ** 2)
            output_dict.update({"kl_loss": kl_loss})
            # output_dict.update({
            #     'z_mu': z_mu,
            #     'z_sigma': z_sigma})

        print(x.shape, z.shape)
        logits = self.forward_decoder(z_sampled, x.shape)

        # print(x.shape,z.shape,z_sampled.shape,logits.shape)

        output_dict.update({"logits": logits})
        output_dict.update({"middd": z_sampled})

        if not self.training:
            pred = logits.argmax(dim=-1).detach().cuda()
            output_dict["sem_pred"] = pred
            pred_iou = deepcopy(pred)

            pred_iou[pred_iou != 17] = 1
            pred_iou[pred_iou == 17] = 0
            output_dict["iou_pred"] = pred_iou

        return output_dict
        # loss, kl, rec = self.loss(logits, x, z_mu, z_sigma)
        # return loss, kl, rec

    def encode(self, x, **kwargs):
        z, shapes = self.forward_encoder(x)
        z_sampled, z_mu, z_sigma = self.sample_z(z)
        return z_sampled

    def generate(self, z, input_shape):
        logits = self.forward_decoder(z, input_shape)
        return {"logits": logits}

    def generate_vq(self, z, shapes, input_shape):
        z_sampled, _, _ = self.vqvae(z, is_voxel=False)
        logits = self.forward_decoder(z_sampled, shapes, input_shape)
        return {"logits": logits}


class AttentionResidualBlock(nn.Module):
    def __init__(self, n_hiddens, axial_block_type="thw"):
        super().__init__()
        self.block = nn.Sequential(
            # nn.BatchNorm3d(n_hiddens),
            # nn.ReLU(),
            nn.Conv3d(n_hiddens, n_hiddens // 2, kernel_size=(1, 1, 1), stride=1, padding=0, bias=False),
            # nn.BatchNorm3d(n_hiddens // 2),
            Normalize(n_hiddens // 2),
            nn.ReLU(),
            nn.Conv3d(n_hiddens // 2, n_hiddens, kernel_size=(1, 1, 1), stride=1, padding=0, bias=False),
            # nn.BatchNorm3d(n_hiddens),
            Normalize(n_hiddens),
            # nn.ReLU(), # delete 1421 for test_VAEdec.py
            self.get_axialblock(n_hiddens, axial_block_type)
            # AxialBlock(n_hiddens, 2)
        )

    def get_axialblock(self, n_hiddens, axial_block_type):
        if axial_block_type == "thw":
            return AxialBlock(n_hiddens, 2)
        elif axial_block_type == "hw":
            return AxialBlock_wh(n_hiddens, 2)

    def forward(self, x):
        return x + self.block(x)


class AxialBlock_wh(nn.Module):
    def __init__(self, n_hiddens, n_head):
        super().__init__()
        kwargs = dict(
            shape=(0,) * 3, dim_q=n_hiddens, dim_kv=n_hiddens, n_head=n_head, n_layer=1, causal=False, attn_type="axial"
        )
        self.attn_w = MultiHeadAttention(attn_kwargs=dict(axial_dim=-2), **kwargs)
        self.attn_h = MultiHeadAttention(attn_kwargs=dict(axial_dim=-3), **kwargs)
        # self.attn_t = MultiHeadAttention(attn_kwargs=dict(axial_dim=-4),
        #                                  **kwargs)

    def forward(self, x):
        x = shift_dim(x, 1, -1)
        x = self.attn_w(x, x, x) + self.attn_h(x, x, x)  # + self.attn_t(x, x, x)
        x = shift_dim(x, -1, 1)
        return x


class Attn_ResBlock_wh(nn.Module):
    def __init__(self, n_hiddens):
        super().__init__()
        self.block = nn.Sequential(Normalize(n_hiddens), AxialBlock_wh(n_hiddens, 2))

    def forward(self, x):
        return x + self.block(x)


class AxialBlock(nn.Module):
    def __init__(self, n_hiddens, n_head):
        super().__init__()
        kwargs = dict(
            shape=(0,) * 3, dim_q=n_hiddens, dim_kv=n_hiddens, n_head=n_head, n_layer=1, causal=False, attn_type="axial"
        )
        self.attn_w = MultiHeadAttention(attn_kwargs=dict(axial_dim=-2), **kwargs)
        self.attn_h = MultiHeadAttention(attn_kwargs=dict(axial_dim=-3), **kwargs)
        self.attn_t = MultiHeadAttention(attn_kwargs=dict(axial_dim=-4), **kwargs)

    def forward(self, x):
        x = shift_dim(x, 1, -1)
        x = self.attn_w(x, x, x) + self.attn_h(x, x, x) + self.attn_t(x, x, x)
        x = shift_dim(x, -1, 1)
        return x


@MODELS.register_module()
class Decoder3D_withT(nn.Module):
    def __init__(
        self,
        z_channels,
        n_hiddens,
        n_res_layers,
        upsample,
        final_channels=128,
        ch_mult=[4, 2, 1],
        axial_block_type="thw",
    ):
        super().__init__()
        n_hidden_input = n_hiddens * ch_mult[0]
        self.convz_in = nn.Conv3d(z_channels, n_hidden_input, kernel_size=1, stride=1, bias=False, padding=0)
        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(n_hidden_input, axial_block_type) for _ in range(n_res_layers)],
            # nn.BatchNorm3d(n_hidden_input),
            Normalize(n_hidden_input),
            nn.ReLU(),
        )

        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()
        self.convts = nn.ModuleList()
        self.attn_res = nn.ModuleList()
        in_ch_mult = (ch_mult[0],) + ch_mult  # (4,)+ch_mult
        for i in range(max_us):
            in_channels = n_hiddens * in_ch_mult[i]
            out_channels = final_channels if i == max_us - 1 else n_hiddens * ch_mult[i]
            # convt = nn.ConvTranspose3d(n_hiddens, out_channels, kernel_size=(1,2,2),
            #                             stride=(1,2,2), bias=False,
            #                             padding=0)
            convt = nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                bias=False,
                padding=(0, 1, 1),
                output_padding=(0, 1, 1),
            )
            if i != max_us - 1:
                attn_res = nn.Sequential(
                    *[AttentionResidualBlock(out_channels, axial_block_type) for _ in range(n_res_layers)]
                )  # modify 1421
                # attn_res = nn.Sequential(*[AttentionResidualBlock(out_channels) for _ in range(n_res_layers)],
                #                             Normalize(n_hidden_input),
                #                             nn.ReLU())
                self.attn_res.append(attn_res)
            self.convts.append(convt)
            n_times_upsample -= 1

    def forward(self, x):
        h = self.convz_in(x)
        h = self.res_stack(h)
        for i, convt in enumerate(self.convts):
            # print(h.shape)
            h = convt(h)
            if i < len(self.convts) - 1:
                h = self.attn_res[i](h)
                h = F.relu(h)  # add 1421
        return h


@MODELS.register_module()
class Encoder2D_new2(BaseModule):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    print("[*] Enc has Attn at i_level, i_block: %d, %d" % (i_level, i_block))
                    # attn.append(AttnBlock(block_in))
                    attn.append(Attn_ResBlock_wh(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        self.mid.attn_1 = Attn_ResBlock_wh(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, 2 * z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        # x: bs, F, H, W, D
        shapes = []
        temb = None

        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):

            for i_block in range(self.num_res_blocks):
                # h = self.down[i_level].block[i_block](hs[-1], temb)
                h = self.down[i_level].block[i_block](h, temb)

                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                # hs.append(h)
            if i_level != self.num_resolutions - 1:
                shapes.append(h.shape[-2:])
                # hs.append(self.down[i_level].downsample(hs[-1]))
                h = self.down[i_level].downsample(h)

        # middle
        # h = hs[-1]
        #
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        # h = nonlinearity(h)
        h = F.relu(h)
        h = self.conv_out(h)
        return h, shapes


if __name__ == "__main__":
    # test encoder
    # import torch
    # encoder = Encoder2D(in_channels=128, ch=8, out_ch=8, ch_mult=(1,2,4), num_res_blocks=2, resolution=200,attn_resolutions=(50,), z_channels=4, double_z=False)
    # # #decoder = Decoder3D()
    # # decoder = Decoder2D(in_channels=3, ch=64, out_ch=3, ch_mult=(1,2,4,8), num_res_blocks=2, resolution=200,attn_resolutions=(100,50), z_channels=64, give_pre_end=False)
    # decoder = Decoder2D_withT(in_channels=128, ch=8, out_ch=128, ch_mult=(1,2,4), num_res_blocks=2, resolution=200,attn_resolutions=(50,), z_channels=4, t_shape =10, give_pre_end=False)

    # #b 5, t 10
    # input = torch.randn((50,128,200,200))
    # z,shapes = encoder(input)
    # print(z.shape)
    # print(shapes)
    # rec =decoder(z,shapes)
    # print(rec.shape)
    # import pdb; pdb.set_trace()
    # attn3d = AttnBlock3D(in_channels=12)
    # input = torch.randn((40,12,50,50))
    # output = attn3d(input,t_shape=8)
    # print(output.shape)
    pass

    _dim_ = 16
    expansion = 8
    # base_channel = 64
    base_channel = 4
    n_e_ = 512
    return_len_ = 10
    model_dict = dict(
        encoder_cfg=dict(
            type="Encoder2D_new",
            ch=base_channel * 8,
            out_ch=base_channel * 2,  # useless
            ch_mult=(1, 2, 4),
            num_res_blocks=4,
            attn_resolutions=(50,),
            dropout=0.0,
            resamp_with_conv=True,
            in_channels=_dim_ * expansion,
            resolution=200,
            z_channels=base_channel,
            double_z=True,
        ),
        decoder_cfg=dict(
            type="Decoder3D_withT",
            z_channels=base_channel,
            ch_mult=(4, 2, 1),
            n_hiddens=base_channel * 8,
            n_res_layers=4,
            upsample=(1, 4, 4),
        ),
        num_classes=18,
        expansion=expansion,
        # vqvae_cfg=dict(
        #     type='VectorQuantizer',
        #     n_e = n_e_,
        #     e_dim = 256,
        #     beta = 1.,
        #     z_channels = base_channel,
        #     use_voxel=True)
    )

    VAE_model = VAERes2D_DwT(**model_dict)

    input = torch.randint(0, 18, (2, 10, 200, 200, 16))
    out = VAE_model(input)

    encoder = VAE_model.encoder
    n_parameters = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"encoder p: {n_parameters}")

    decoder = VAE_model.decoder
    n_parameters = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"decoder p: {n_parameters}")

    # print(out.shape)
