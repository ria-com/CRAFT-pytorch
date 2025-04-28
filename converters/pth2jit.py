#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_____________________________________________________________________________

This file contains converter from torch .pth model to torchscript (jit) .pt model for Triton

EXAMPLE RUN:
 CUDA_VISIBLE_DEVICES=0 python3.12 ./pth2jit.py --weight_craft_path=../../data/models/craft/craft_mlt_25k_2020-02-16.pth
 CUDA_VISIBLE_DEVICES=0 python3.12 ./pth2jit.py --weight_craft_path=../../data/models/craft/craft_mlt_25k_2020-02-16.pth --weight_refine_net_path=../../data/models/craft/craft_refiner_CTW1500_2020-02-16.pth
_____________________________________________________________________________
"""
import sys
import torch
import os
import argparse
from pathlib import Path
import torch.backends.cudnn as cudnn
from utils import initial_logger, copyStateDict

sys.path.append("../")
from craft import CRAFT

logger = initial_logger()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def convert_net_to_jit(net, input_shape=(1, 3, 768, 768),
                       output_path='../model_repository/detec_pt/1/detec_pt.pt'):
    # Prepare output name
    # Set output path for .pt file
    output_path = Path(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # An example input you would normally provide to your model's forward() method.
    x = torch.randn(*input_shape).to(device)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(net, x)

    # Save the TorchScript model
    traced_script_module.save(output_path)
    logger.info("Convert model detec from .pth to .pt (JIT) completed!")
    y = net(x)
    return y[0].shape, y[1].shape


def convert_refine_to_jit(net, input_shape_1=(1, 384, 384, 2),
                        input_shape_2=(1, 32, 384, 384),
                        output_path='../model_repository/refine_pt/1/refine_pt.pt'):
    # Prepare output name
    # Set output path for .pt file
    output_path = Path(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)


    # An example input you would normally provide to your model's forward() method.
    x1 = torch.randn(*input_shape_1).to(device)
    x2 = torch.randn(*input_shape_2).to(device)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(net, (x1, x2))

    # Save the TorchScript model
    traced_script_module.save(output_path)
    logger.info("Convert model refine from .pth to .pt (JIT) completed!")


def convert_detec(weight_craft_path, weight_refine_net_path):
    # load and  net
    net = CRAFT()
    logger.info(f"Converting to JIT (.pt) craft model from : {weight_craft_path}")
    if device.type == 'cuda':
        net.load_state_dict(copyStateDict(torch.load(weight_craft_path)))
    else:
        net.load_state_dict(copyStateDict(torch.load(weight_craft_path, map_location='cpu')))
    if device.type == 'cuda':
        net = net.cuda()
        cudnn.benchmark = False
    net.eval()
    y_shape, feature_shape = convert_net_to_jit(net, input_shape=(1, 3, 768, 768),
                                                output_path='../model_repository/detec_pt/1/detec_pt.pt')

    # load refinenet
    if weight_refine_net_path:
        from refinenet import RefineNet
        refine_net = RefineNet()
        logger.info(f"Converting to JIT (.pt) refine net model from : {weight_refine_net_path}")
        if device.type == 'cuda':
            refine_net.load_state_dict(copyStateDict(torch.load(weight_refine_net_path)))
            refine_net = refine_net.cuda()
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(weight_refine_net_path, map_location='cpu')))

        refine_net.eval()
        convert_refine_to_jit(refine_net, input_shape_1=y_shape,
                              input_shape_2=feature_shape,
                              output_path='../model_repository/refine_pt/1/refine_pt.pt')



def main():
    parser = argparse.ArgumentParser(description="Convert torch model (.pth) into torchscript model (.pt)")
    parser.add_argument("--weight_craft_path",
                        required=True, help="Path to input model weights craft")
    parser.add_argument("--weight_refine_net_path",
                        required=False, help="Path to input model weights craft")

    args=parser.parse_args()

    convert_detec(**vars(args))


if __name__ == '__main__':
    main()
