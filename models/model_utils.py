"""
# Description: utils functions that use for model
"""
from models.darknet2pytorch import Darknet

import torch


def create_model(config):
    """Create model based on architecture name"""
    if (config.arch == 'darknet') and (config.cfgfile is not None):
        print('using darknet')
        model_data = Darknet(
            cfgfile=config.cfgfile, use_giou_loss=config.use_giou_loss, use_diou_loss=config.use_diou_loss)
    else:
        assert False, 'Undefined model backbone'

    return model_data


def get_num_parameters(model_data):
    """Count number of trained parameters of the model"""
    if hasattr(model_data, 'module'):
        num_parameters = sum(p.numel() for p in model_data.module.parameters() if p.requires_grad)
    else:
        num_parameters = sum(p.numel() for p in model_data.parameters() if p.requires_grad)

    return num_parameters


if __name__ == '__main__':
    import argparse

    from torchsummary import summary
    from easydict import EasyDict as eDict

    parser = argparse.ArgumentParser(description='Complexer YOLO Implementation')
    parser.add_argument('-a', '--arch', type=str, default='darknet', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--cfgfile', type=str, default='../config/cfg/complex_yolov4.cfg', metavar='PATH',
                        help='The path for cfgfile (only for darknet)')

    configs = eDict(vars(parser.parse_args()))

    configs.device = torch.device('cuda:1')

    model = create_model(configs).to(device=configs.device)
    sample_input = torch.randn((1, 3, 608, 608)).to(device=configs.device)
    # summary(model.cuda(), (3, 608, 608))
    output = model(sample_input, targets=None)
    print(output.size())
