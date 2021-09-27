import pdb

import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
import torch.nn.functional as F


class VGGPyramid(nn.Module):
    def __init__(self, train=False):
        super().__init__()
        self.n_levels = 5
        source_model = models.vgg16(pretrained=True)

        modules = OrderedDict()
        tmp = []
        n_block = 0
        first_relu = False

        for c in source_model.features.children():
            if (isinstance(c, nn.ReLU) and not first_relu) or (isinstance(c, nn.MaxPool2d)):
                first_relu = True
                tmp.append(c)
                modules['level_' + str(n_block)] = nn.Sequential(*tmp)
                for param in modules['level_' + str(n_block)].parameters():
                    param.requires_grad = train

                tmp = []
                n_block += 1
            else:
                tmp.append(c)

            if n_block == self.n_levels:
                break

        self.__dict__['_modules'] = modules

    def forward(self, x):
        outputs = []

        for layer_n in range(0, self.n_levels):
            x = self.__dict__['_modules']['level_' + str(layer_n)](x)
            outputs.append(x)
        return outputs

