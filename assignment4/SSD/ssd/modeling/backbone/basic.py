import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models 

class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
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
            nn.Conv2d(self.output_channels[-2], self.output_channels[-2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.output_channels[-2], self.output_channels[-1], kernel_size=3, stride=1, padding=1)
        ))

    
    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        for bank in self.layers:
            x = bank(x)
            out_features.append(x)
        for idx, feature in enumerate(out_features):
            print(feature.shape[1:])
            out_channel = feature.shape[1]
            feature_map_size = feature.shape[2]
            expected_shape = (out_channel, feature_map_size, feature_map_size)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)

