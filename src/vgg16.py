import sys
import os
import torch
import torch.nn as nn
import torchvision.models as models


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.model = "VGG16"

        self.model = models.vgg16(pretrained=True)

        for params in self.model.parameters():
            params.requires_grad = False

        self.model = nn.Sequential(list(self.model.children())[0][0:13])

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    vgg16 = VGG16()
    # print(vgg16.model)
