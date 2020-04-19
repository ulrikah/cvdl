import torch
from math import sqrt
from itertools import product


class PriorBox:
    def __init__(self, cfg):
        self.image_size = cfg.INPUT.IMAGE_SIZE
        prior_config = cfg.MODEL.PRIORS
        self.feature_maps = prior_config.FEATURE_MAPS
        self.min_sizes = prior_config.MIN_SIZES
        self.max_sizes = prior_config.MAX_SIZES
        self.strides = prior_config.STRIDES
        self.aspect_ratios = prior_config.ASPECT_RATIOS
        self.clip = prior_config.CLIP

    def __call__(self):
        """Generate SSD Prior Boxes.
            It returns the center, height and width of the priors. The values are relative to the image size
            Returns:
                priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
                    are relative to the image size.
        """
        priors = []
        for k, f in enumerate(self.feature_maps):
            if isinstance(self.image_size, list):
                image_size_w, image_size_h = self.image_size
            elif isinstance(self.image_size, int):
                image_size_w = image_size_h = self.image_size
            scale_w  = image_size_w / self.strides[k][0]
            scale_h  = image_size_h / self.strides[k][1]
            for i in range(f[1]): # height
                for j in range(f[0]): # width
                    # unit center x,y
                    cx = (j + 0.5) / scale_w
                    cy = (i + 0.5) / scale_h

                    # small sized square box
                    size = self.min_sizes[k]
                    w = size / image_size_w
                    h = size / image_size_h
                    priors.append([cx, cy, w, h])

                    # big sized square box
                    size = sqrt(self.min_sizes[k] * self.max_sizes[k])
                    w = size / image_size_w
                    h = size / image_size_h
                    priors.append([cx, cy, w, h])

                    # change h/w ratio of the small sized box
                    size = self.min_sizes[k]
                    w = size / image_size_w
                    h = size / image_size_h
                    for ratio in self.aspect_ratios[k]:
                        ratio = sqrt(ratio)
                        priors.append([cx, cy, w * ratio, h / ratio])
                        priors.append([cx, cy, w / ratio, h * ratio])

        priors = torch.tensor(priors)
        if self.clip:
            priors.clamp_(max=1, min=0)
        return priors
