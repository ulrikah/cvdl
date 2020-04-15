import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models

class ResNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        image_size = cfg.INPUT.IMAGE_SIZE
        self.output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS


        # TO DO: 
        # - option for pretraining
        # - option to choose other ResNet variants

        backbone = models.resnet34(pretrained=False)
        # self.output_channels = [256, 512, 512, 256, 256, 256]

        # we don't use the two last layers of ResNet
        self.layers = nn.Sequential(*list(backbone.children())[:7])

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            print(f"Output shape of layer {i}: {x.shape[1:]}")
            features.append(x)
        return tuple(features)
