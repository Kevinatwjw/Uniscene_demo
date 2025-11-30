# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

############
import math

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed

from .bev_cod import BEV_concat_net_s, BEV_condition_net

# from bev_cod import BEV_condition_net,BEV_concat_net
# from embedder import get_embedder


XYZ_MIN = [-200, -300, -20]
XYZ_RANGE = [350, 650, 80]


def normalizer(mode, data):
    if mode == "cxyz" or mode == "all-xyz":
        # data in format of (N, 4, 3):
        mins = torch.as_tensor(XYZ_MIN, dtype=data.dtype, device=data.device)[None, None]
        divider = torch.as_tensor(XYZ_RANGE, dtype=data.dtype, device=data.device)[None, None]
        data = (data - mins) / divider
    elif mode == "owhr":
        raise NotImplementedError(f"wait for implementation on {mode}")
    else:
        raise NotImplementedError(f"not support {mode}")
    return data


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings.

        :param t: a 1-D Tensor of N indices, one per batch element. These may
            be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations.

    Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """Drops labels to enable classifier-free guidance."""
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class BEVDropout_layer(nn.Module):
    """Embeds class labels into vector representations.

    Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, dropout_prob, use_3d=False):
        super().__init__()
        # use_cfg_embedding = dropout_prob > 0
        # self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        # self.num_classes = num_classes
        if use_3d:
            self.maxpool = nn.MaxPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4), padding=(0, 0, 0))
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        self.dropout_prob = dropout_prob
        if dropout_prob > 0:
            print("Use BEV Dropout!")

    def token_drop(self, BEV_layout):
        """Drops labels to enable classifier-free guidance."""
        if torch.rand(1) < self.dropout_prob:
            BEV_null = -torch.ones_like(BEV_layout, device=BEV_layout.device)
            BEV_layout = BEV_null
        # drop_mask = torch.rand_like(BEV_layout,device=BEV_layout.device) < self.dropout_prob
        # BEV_layout[drop_mask] =-1
        return BEV_layout

    def forward(self, BEV_layout):
        use_dropout = self.dropout_prob > 0

        BEV_layout = self.maxpool(BEV_layout)
        if self.training and use_dropout:
            BEV_layout = self.token_drop(BEV_layout)
        # embeddings = self.embedding_table(labels)
        return BEV_layout


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU()  # lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """The final layer of DiT."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class BEVNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=18, out_channels=64, kernel_size=3, stride=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, ceil_mode=False, padding=1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        return x


class MLP_meta(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout_prob = dropout_prob
        print(f"MLP_meta: {input_size}, {hidden_size}, {output_size}")

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        if self.training and self.dropout_prob > 0:
            if torch.rand(1) < self.dropout_prob:
                x_null = torch.zeros_like(x, device=x.device)
                x = x_null
        return x


class OccDiT(nn.Module):
    """Diffusion model with a Transformer backbone."""

    def __init__(
        self,
        input_size=50,
        patch_size=2,
        in_channels=4,
        hidden_size=256,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        bev_dropout_prob=0,
        # num_classes=1000,
        bev_in_ch=1,
        bev_out_ch=1,
        meta_num=1,
        learn_sigma=True,
        use_label=False,
        use_meta=False,
        use_bev_concat=True,
        direct_concat=False,
        Tframe=6,
        temp_attn=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels + bev_out_ch if use_bev_concat else in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_label = use_label
        self.use_meta = use_meta
        self.depth = depth
        self.Tframe = Tframe
        self.temp_attn = temp_attn
        self.use_bev_concat = use_bev_concat
        self.x_embedder = PatchEmbed(input_size, patch_size, self.in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        if use_label:
            self.y_embedder = BEV_condition_net()
        if use_meta:
            print(f"Use Meta embedding! meta_num:{meta_num}")
            # self.meta_embedder = nn.Linear(meta_num,hidden_size)
            # self.meta_embedder = MLP_meta(meta_num,64,hidden_size,bev_dropout_prob)
            self.meta_embedder = MLP_meta(meta_num, int(hidden_size / 2), hidden_size, bev_dropout_prob)
        if self.use_bev_concat:
            if direct_concat == False:
                # self.bev_concat = BEV_concat_net(BEV_in_ch=bev_in_ch,BEV_out_ch=bev_out_ch)
                self.bev_concat = BEV_concat_net_s(BEV_in_ch=bev_in_ch, BEV_out_ch=bev_out_ch)
            else:
                print("Use MaxPool3d!")
                # self.bev_concat = nn.MaxPool3d(kernel_size=(1,4,4), stride=(1,4,4), padding=(0,0,0))
                self.bev_concat = BEVDropout_layer(bev_dropout_prob, use_3d=True)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed_m = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.temp_embed = nn.Parameter(torch.zeros(1, Tframe, hidden_size), requires_grad=False)

        # self.register_buffer("pos_embed_temporal", self.get_temporal_pos_embed())

        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed_m.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed_m.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        temp_embed = get_1d_sincos_pos_embed(self.hidden_size, self.Tframe)
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # if self.use_label:
        #     nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y, meta=None):
        """
        x: [N,C,T,H,W]
        t: [N,]
        y: [N,T,Cb,Hb,Wb] (BEV)
        meta: [N,m]
        """
        x = x.permute(0, 2, 1, 3, 4)
        # print(x.shape,y.shape)
        if self.use_bev_concat:
            x = torch.cat([x, self.bev_concat(y)], dim=2)
        # print(x.shape)
        B, T, C, H, W = x.shape
        S = 625
        x = x.reshape(B * T, C, H, W)
        x = self.x_embedder(x)  # B*T,S,D

        x = x + self.pos_embed_m

        if self.temp_attn == True:
            x = rearrange(x, "(B T) S D -> (B S) T D", T=T, S=S)
            x = x + self.temp_embed
            x = rearrange(x, "(B S) T D -> (B T) S D", T=T, S=S)

        # x = x.reshape(B,-1,self.hidden_size) # B T*patch_num D
        # print(x.shape)
        t = self.t_embedder(t)  # (B, D)
        # print(t.shape)
        if self.use_meta:
            pts_num = meta
            meta_embd = self.meta_embedder(pts_num)
            c = t + meta_embd  # B D
        else:
            c = t

        c_s = c.repeat(1, 1, T)
        c_s = c_s.reshape(-1, self.hidden_size)
        c_t = c.repeat(1, 1, S)
        c_t = c_t.reshape(-1, self.hidden_size)

        for i, block in enumerate(self.blocks):
            if self.temp_attn == True:
                if i % 2 == 0:  # (B*T,S,D) spatial
                    if x.shape[0] == B * S:
                        x = rearrange(x, "(B S) T D -> (B T) S D", T=T, S=S)
                    x = block(x, c_s)
                else:  # (B*S,T,D) temporal
                    if x.shape[0] == B * T:
                        x = rearrange(x, "(B T) S D -> (B S) T D", T=T, S=S)
                    x = block(x, c_t)
            else:
                x = block(x, c_s)  # only spatial

        if x.shape[0] == B * S:
            x = rearrange(x, "(B S) T D -> (B T) S D", T=T, S=S)
        x = self.final_layer(x, c_s)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        # print(x.shape)
        x = x.reshape(B, T, self.out_channels, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        return x

    def forward_with_cfg(self, x, t, y, meta=None, cfg_scale=1.0):
        """Forward pass of DiT, but also batches the unconditional forward pass
        for classifier-free guidance."""
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y, meta)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class DiT_WorldModel(nn.Module):
    """Diffusion model with a Transformer backbone."""

    def __init__(
        self,
        input_size=50,
        patch_size=2,
        in_channels=4,
        hidden_size=256,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        bev_dropout_prob=0,
        # num_classes=1000,
        bev_in_ch=1,
        bev_out_ch=1,
        meta_num=1,
        learn_sigma=True,
        use_label=False,
        use_meta=False,
        use_bev_concat=True,
        direct_concat=False,
        T_pred=6,
        T_condition=1,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels + bev_out_ch if use_bev_concat else in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_label = use_label
        self.use_meta = use_meta
        self.depth = depth
        self.T_pred = T_pred
        self.T_condition = T_condition
        self.Tframe = T_pred + T_condition
        self.use_bev_concat = use_bev_concat
        self.x_embedder = PatchEmbed(input_size, patch_size, self.in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        if use_label:
            self.y_embedder = BEV_condition_net()
        if use_meta:
            print(f"Use Meta embedding! meta_num:{meta_num}")
            self.meta_embedder = MLP_meta(meta_num, int(hidden_size / 2), hidden_size, bev_dropout_prob)
        if self.use_bev_concat:
            if direct_concat == False:
                # self.bev_concat = BEV_concat_net(BEV_in_ch=bev_in_ch,BEV_out_ch=bev_out_ch)
                self.bev_concat = BEV_concat_net_s(BEV_in_ch=bev_in_ch, BEV_out_ch=bev_out_ch)
            else:
                print("Use MaxPool3d!")
                # self.bev_concat = nn.MaxPool3d(kernel_size=(1,4,4), stride=(1,4,4), padding=(0,0,0))
                self.bev_concat = BEVDropout_layer(bev_dropout_prob, use_3d=True)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed_m = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.temp_embed = nn.Parameter(torch.zeros(1, self.Tframe, hidden_size), requires_grad=False)

        # self.register_buffer("pos_embed_temporal", self.get_temporal_pos_embed())

        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed_m.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed_m.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        temp_embed = get_1d_sincos_pos_embed(self.hidden_size, self.Tframe)
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # if self.use_label:
        #     nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, x_ref, y, meta=None):
        """
        x: [N,C,Tp,H,W]
        t: [N,]
        x_ref: [N,C,Tc,H,W]
        y: [N,Tp,Cb,Hb,Wb] (BEV)
        meta: [N,m]
        """
        x = x.permute(0, 2, 1, 3, 4)
        x_ref = x_ref.permute(0, 2, 1, 3, 4)

        # print(x.shape,y.shape)
        if self.use_bev_concat:
            x = torch.cat([x, self.bev_concat(y)], dim=2)
            y_ref = torch.zeros_like(x_ref[:, :, 0:1], device=x.device)
            # print(x_ref.shape,y_ref.shape)

            x_ref = torch.cat([x_ref, y_ref], dim=2)  # B Tc C H W
            # print(x.shape,x_ref.shape)

        x = torch.cat([x_ref, x], dim=1)  # B (Tp + Tc) C H W
        # print(x.shape)
        B, T, C, H, W = x.shape
        S = 625
        x = x.reshape(B * T, C, H, W)
        x = self.x_embedder(x)  # B*T,S,D

        x = x + self.pos_embed_m
        x = rearrange(x, "(B T) S D -> (B S) T D", T=T, S=S)
        x = x + self.temp_embed
        x = rearrange(x, "(B S) T D -> (B T) S D", T=T, S=S)

        # x = x.reshape(B,-1,self.hidden_size) # B T*patch_num D
        # print(x.shape)
        t = self.t_embedder(t)  # (B, D)
        # print(t.shape)
        if self.use_meta:
            pts_num = meta
            meta_embd = self.meta_embedder(pts_num)
            c = t + meta_embd  # B D
        else:
            c = t

        c_s = c.repeat(1, 1, T)
        c_s = c_s.reshape(-1, self.hidden_size)
        c_t = c.repeat(1, 1, S)
        c_t = c_t.reshape(-1, self.hidden_size)

        for i, block in enumerate(self.blocks):
            if i % 2 == 0:  # (B*T,S,D) spatial
                if x.shape[0] == B * S:
                    x = rearrange(x, "(B S) T D -> (B T) S D", T=T, S=S)
                x = block(x, c_s)
            else:  # (B*S,T,D) temporal
                if x.shape[0] == B * T:
                    x = rearrange(x, "(B T) S D -> (B S) T D", T=T, S=S)
                x = block(x, c_t)

        x = rearrange(x, "(B S) T D -> (B T) S D", T=T, S=S)
        x = self.final_layer(x, c_s)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        # print(x.shape)
        x = x.reshape(B, T, self.out_channels, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        x = x[:, :, self.T_condition :]
        return x

    def forward_with_cfg(self, x, t, x_ref, y, meta=None, cfg_scale=1.0):
        """Forward pass of DiT, but also batches the unconditional forward pass
        for classifier-free guidance."""
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, x_ref, y, meta)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    # grid = grid.reshape([2, 1, grid_size,grid_size])
    grid = grid.reshape([2, grid_size * grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(length, dtype=np.float32)  # [..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):  # hidden_size=embed_dim 1152 pos=256
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)

    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    # print("omega",omega.shape) #288
    # print("pos",pos.shape)

    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

    return emb  #  256 576
