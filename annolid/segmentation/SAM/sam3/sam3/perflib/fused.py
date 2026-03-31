# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import torch

addmm_act_op = torch.ops.aten._addmm_activation


def addmm_act(activation, linear, mat1):
    if torch.is_grad_enabled():
        raise ValueError("Expected grad to be disabled.")
    out_dtype = mat1.dtype
    acc_dtype = (
        torch.bfloat16
        if mat1.device.type == "cuda"
        else out_dtype
    )
    self = linear.bias.detach()
    mat2 = linear.weight.detach()
    self = self.to(acc_dtype)
    mat1 = mat1.to(acc_dtype)
    mat2 = mat2.to(acc_dtype)
    mat1_flat = mat1.view(-1, mat1.shape[-1])
    if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
        y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=False)
        return y.view(mat1.shape[:-1] + (y.shape[-1],)).to(out_dtype)
    if activation in [torch.nn.functional.gelu, torch.nn.GELU]:
        y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=True)
        return y.view(mat1.shape[:-1] + (y.shape[-1],)).to(out_dtype)
    raise ValueError(f"Unexpected activation {activation}")
