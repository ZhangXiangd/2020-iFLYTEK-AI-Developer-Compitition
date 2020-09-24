import torch
import torchvision
from torchvision.models.resnet import ResNet, Bottleneck

def _resnext(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    state_dict = torch.load('./pretrained_models/ig_resnext101_32x8-c38310e5.pth')
    model.load_state_dict(state_dict)
    return model

def resnext101_32x8d_wsl(pretrained=False, **kwargs):
    """Constructs a ResNeXt-101 32x8 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnext('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained, progress=True, **kwargs)