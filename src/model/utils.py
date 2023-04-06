import PIL
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F


class MaskMSELoss(Module):
    def __init__(self, alpha=1, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor, mask: Tensor):
        mask_loss = F.mse_loss(
            input[mask == 1], target[mask == 1], reduction="sum")
        non_mask_loss = F.mse_loss(
            input[mask == 0], target[mask == 0], reduction="sum")
        return (self.alpha * mask_loss + non_mask_loss) / torch.numel(mask)


@torch.no_grad()
def convert_fourchannel_unet(oldunet, newunet):
    # 1. replace conv_in weight since they have different in_channels
    old_conv_in = oldunet.conv_in
    oldunet.conv_in = newunet.conv_in
    # 2. put old conv_in weight back
    oldunet.conv_in.weight[:, :4, :, :] = old_conv_in.weight
    oldunet.conv_in.bias = old_conv_in.bias
    # 3. copy other weights to newunet
    convert_single_cond_unet(oldunet, newunet)


@torch.no_grad()
def convert_single_cond_unet(oldunet, newunet):
    if type(newunet.config.cross_attention_dim) == dict:
        #! convert old unet to MultiCondition2DUnet
        # 1. replace resnets and other modules not cross-attention related
        newunet.load_state_dict(oldunet.state_dict(), strict=False)
        # 2. replace old text cross-attention modules into new model attentions['text'] module
        if "text" in newunet.config.cross_attention_dim:
            for i in range(len(oldunet.down_blocks)):
                if hasattr(oldunet.down_blocks[i], "attentions"):
                    newunet.down_blocks[i].attentions["text"].load_state_dict(
                        oldunet.down_blocks[i].attentions.state_dict()
                    )
                else:
                    newunet.down_blocks[i].load_state_dict(
                        oldunet.down_blocks[i].state_dict()
                    )
            newunet.mid_block.attentions["text"].load_state_dict(
                oldunet.mid_block.attentions.state_dict()
            )
            for i in range(len(oldunet.up_blocks)):
                if hasattr(oldunet.up_blocks[i], "attentions"):
                    newunet.up_blocks[i].attentions["text"].load_state_dict(
                        oldunet.up_blocks[i].attentions.state_dict()
                    )
                else:
                    newunet.up_blocks[i].load_state_dict(
                        oldunet.up_blocks[i].state_dict())
    else:
        #! convert old unet to 2DConditionUNet
        newunet.load_state_dict(oldunet.state_dict(), strict=True)
    # 3. we don't replace attentions['char'] module because it is not present in old model
