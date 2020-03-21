import torch
import torch.nn as nn
import torch.nn.functional as F

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


        # VGG
        # 38 x 38
        bank1 = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.output_channels[0], kernel_size=3, padding=1, stride=2)
        )
        # 19 x 19
        bank2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(self.output_channels[0], self.output_channels[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.output_channels[0], self.output_channels[1], kernel_size=3, stride=2, padding=1)
        )
        # 9 x 9
        bank3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(self.output_channels[1], self.output_channels[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.output_channels[1], self.output_channels[2], kernel_size=3, stride=2, padding=1)
        )
        # 5 x 5
        bank4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(self.output_channels[2], self.output_channels[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.output_channels[2], self.output_channels[3], kernel_size=3, stride=2, padding=1)
        )
        # 3 x 3
        bank5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(self.output_channels[3], self.output_channels[3], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.output_channels[3], self.output_channels[4], kernel_size=3, stride=2, padding=1)
        )
        # 1 x 1
        bank6 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(self.output_channels[4], self.output_channels[4], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.output_channels[4], self.output_channels[5], kernel_size=3, stride=1, padding=0)
        )

        self.layers = nn.ModuleList([bank1, bank2, bank3, bank4, bank5, bank6])
    
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
            out_channel = feature.shape[1]
            feature_map_size = feature.shape[2]
            expected_shape = (out_channel, feature_map_size, feature_map_size)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)

