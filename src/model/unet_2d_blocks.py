import numpy as np
import torch
from torch import nn

from diffusers.models.attention import AttentionBlock, Transformer2DModel
from diffusers.models.resnet import Downsample2D, FirDownsample2D, FirUpsample2D, ResnetBlock2D, Upsample2D
from diffusers.models.unet_2d_blocks import get_down_block, get_up_block, UNetMidBlock2DCrossAttn, CrossAttnDownBlock2D, CrossAttnUpBlock2D

from typing import List, Tuple, Union, Optional, Dict


def multicond_get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels: Union[Dict[str, int], int],
    resnet_groups=None,
    cross_attention_dim: Union[Dict[str, int], int] = None,
    downsample_padding=None,
):
    down_block_type = down_block_type[7:] if down_block_type.startswith(
        "UNetRes") else down_block_type
    # original down block
    if not "Multi" in down_block_type:
        return get_down_block(
            down_block_type=down_block_type,
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            attn_num_head_channels=attn_num_head_channels,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            downsample_padding=downsample_padding
        )
    else:
        if cross_attention_dim is None:
            raise ValueError(
                "cross_attention_dim must be specified for MultiCrossAttnDownBlock2D")
        assert type(cross_attention_dim) == dict and type(
            attn_num_head_channels) == dict
        if down_block_type == "MultiCrossAttnDownBlock2D":
            return MultiCrossAttnDownBlock2D(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                num_layers=num_layers,
                add_downsample=add_downsample,
                resnet_eps=resnet_eps,
                resnet_act_fn=resnet_act_fn,
                attn_num_head_channels=attn_num_head_channels,
                resnet_groups=resnet_groups,
                cross_attention_dim=cross_attention_dim,
                downsample_padding=downsample_padding,
            )
        else:
            raise ValueError(
                "Unkown down block type: {}".format(down_block_type))


def multicond_get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels: Union[Dict[str, int], int],
    resnet_groups=None,
    cross_attention_dim: Union[Dict[str, int], int] = None,
):
    up_block_type = up_block_type[7:] if up_block_type.startswith(
        "UNetRes") else up_block_type
    # original down block
    if not "Multi" in up_block_type:
        return get_up_block(
            up_block_type=up_block_type,
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
        )
    else:
        if cross_attention_dim is None:
            raise ValueError(
                "cross_attention_dim must be specified for MultiCrossAttnDownBlock2D")
        assert type(cross_attention_dim) == dict and type(
            attn_num_head_channels) == dict
        if up_block_type == "MultiCrossAttnUpBlock2D":
            return MultiCrossAttnUpBlock2D(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                num_layers=num_layers,
                prev_output_channel=prev_output_channel,
                add_upsample=add_upsample,
                resnet_eps=resnet_eps,
                resnet_act_fn=resnet_act_fn,
                resnet_groups=resnet_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attn_num_head_channels,
            )
        else:
            raise ValueError("Unkown up block type: {}".format(up_block_type))


class UNetMidBlock2DMultiCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels: dict = {"text": 8, "char": 4},
        attention_type="default",
        output_scale_factor=1.0,
        cross_attention_dim: dict = {"text": 1280, "char": 640},
        **kwargs,
    ):
        super().__init__()
        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels
        resnet_groups = resnet_groups if resnet_groups is not None else min(
            in_channels // 4, 32)
        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = {}
        cond_keys = list(cross_attention_dim.keys())
        assert cond_keys == list(attn_num_head_channels.keys(
        )), "cross_attention_dim/attn_num_head_channels must have same keys"
        attentions = {k: [] for k in cond_keys}

        for _ in range(num_layers):
            for k in cond_keys:
                attentions[k].append(
                    Transformer2DModel(
                        attn_num_head_channels[k],
                        in_channels // attn_num_head_channels[k],
                        in_channels=in_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim[k],
                        norm_num_groups=resnet_groups,
                    )
                )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleDict(
            {k: nn.ModuleList(attns) for k, attns in attentions.items()})
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None):
        hidden_states = self.resnets[0](hidden_states, temb)
        for i in range(len(self.resnets[1:])):
            resnet = self.resnets[i + 1]
            attns = {k: self.attentions[k][i] for k in self.attentions}
            cond_hidden_states = {k: attns[k](hidden_states,
                                              encoder_hidden_states=encoder_hidden_states[k]).sample
                                  for k in attns.keys() if encoder_hidden_states[k] is not None}
            hidden_states = torch.mean(torch.stack(
                list(cond_hidden_states.values())), dim=0)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class MultiCrossAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels: dict = {"text": 8, "char": 2},
        cross_attention_dim: dict = {"text": 768, "char": 32},
        attention_type="default",
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
    ):
        super().__init__()
        resnets = []
        cond_keys = list(cross_attention_dim.keys())
        assert cond_keys == list(attn_num_head_channels.keys(
        )), "cross_attention_dim/attn_num_head_channels must have same keys"
        attentions = {k: [] for k in cond_keys}
        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            for k in cond_keys:
                attentions[k].append(
                    Transformer2DModel(
                        attn_num_head_channels[k],
                        out_channels // attn_num_head_channels[k],
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim[k],
                        norm_num_groups=resnet_groups,
                    )
                )

        self.attentions = nn.ModuleDict(
            {k: nn.ModuleList(attns) for k, attns in attentions.items()})
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None, encoder_hidden_states: Dict = None):
        output_states = ()
        for i in range(len(self.resnets)):
            resnet = self.resnets[i]
            attns = {k: self.attentions[k][i] for k in self.attentions}
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)
                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )

                cond_hidden_states = {
                    k: torch.utils.checkpoint.checkpoint(
                        create_custom_forward(
                            attns[k], return_dict=False), hidden_states,
                        encoder_hidden_states[k]
                    )[0] for k in attns.keys() if encoder_hidden_states[k] is not None
                }
                hidden_states = torch.mean(torch.stack(
                    list(cond_hidden_states.values())), dim=0)
            else:
                hidden_states = resnet(hidden_states, temb)
                cond_hidden_states = {k: attns[k](hidden_states,
                                                  encoder_hidden_states=encoder_hidden_states[k]).sample
                                      for k in attns.keys() if encoder_hidden_states[k] is not None}
                hidden_states = torch.mean(torch.stack(
                    list(cond_hidden_states.values())), dim=0)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states += (hidden_states, )
        return hidden_states, output_states


class MultiCrossAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels: dict = {"text": 8, "char": 4},
        cross_attention_dim: dict = {"text": 1280, "char": 640},
        attention_type="default",
        output_scale_factor=1.0,
        add_upsample=True,
    ):
        super().__init__()
        resnets = []
        cond_keys = list(cross_attention_dim.keys())
        assert cond_keys == list(attn_num_head_channels.keys(
        )), "cross_attention_dim/attn_num_head_channels must have same keys"
        attentions = {k: [] for k in cond_keys}
        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            res_skip_channels = in_channels if (
                i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

            for k in cond_keys:
                attentions[k].append(
                    Transformer2DModel(
                        attn_num_head_channels[k],
                        out_channels // attn_num_head_channels[k],
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim[k],
                        norm_num_groups=resnet_groups,
                    )
                )
        self.attentions = nn.ModuleDict(
            {k: nn.ModuleList(attns) for k, attns in attentions.items()})
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        upsample_size=None
    ):
        for i in range(len(self.resnets)):
            resnet = self.resnets[i]
            attns = {k: self.attentions[k][i] for k in self.attentions}

            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb)
                cond_hidden_states = {
                    k: torch.utils.checkpoint.checkpoint(
                        create_custom_forward(
                            attns[k], return_dict=False), hidden_states,
                        encoder_hidden_states[k]
                    )[0] for k in attns.keys() if encoder_hidden_states[k] is not None
                }
                hidden_states = torch.mean(torch.stack(
                    list(cond_hidden_states.values())), dim=0)
            else:
                hidden_states = resnet(hidden_states, temb)
                cond_hidden_states = {k: attns[k](hidden_states,
                                                  encoder_hidden_states=encoder_hidden_states[k]).sample
                                      for k in attns.keys() if encoder_hidden_states[k] is not None}
                hidden_states = torch.mean(torch.stack(
                    list(cond_hidden_states.values())), dim=0)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states
