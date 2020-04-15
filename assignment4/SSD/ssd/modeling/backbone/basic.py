import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models 

class BasicModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        image_size = cfg.INPUT.IMAGE_SIZE
        self.output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS

        self.layers = nn.ModuleList()

        # VGG
        self.layers.append(nn.Sequential(
            nn.Conv2d(image_channels, 128, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.output_channels[0], kernel_size=3, padding=1, stride=2)
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
            print(f"Output shape of layer {i}: {x.shape[1:]}")
            out_features.append(x)
        return tuple(out_features)

