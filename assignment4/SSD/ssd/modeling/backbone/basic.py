import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models 

class BasicModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        image_size = cfg.INPUT.IMAGE_SIZE
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS

        self.layers = nn.ModuleList()

        # we don't use the FC layer of ResNet
        resnet = models.resnet34()
        self.layers.append(nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        ))

        # detection layers
        for i in range(len(self.output_feature_size) - 2):
            self.layers.append(nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(self.output_channels[i], self.output_channels[i], kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.output_channels[i], self.output_channels[i + 1], kernel_size=3, stride=2, padding=1)
            ))
        self.layers.append(nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(self.output_channels[-2], self.output_channels[-2], kernel_size=2, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.output_channels[-2], self.output_channels[-1], kernel_size=2, stride=1, padding=0)
        ))

    
    def forward(self, x):
        out_features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # print(f"Output shape of layer {i}: {x.shape[1:]} \n Should correspond to feature maps")
            out_features.append(x)
        return tuple(out_features)

