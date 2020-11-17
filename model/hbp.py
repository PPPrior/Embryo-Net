"""HBP (Hierarchical Bilinear Pooling for Fine-Grained Visual Recognition)"""

import torch
import torch.nn as nn
from .vgg import vgg16


class HBP(torch.nn.Module):
    def __init__(self, num_classes=200):
        """Declare all needed layers."""
        nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.features = vgg16(pretrained=True).features
        self.features_conv5_1 = nn.Sequential(*list(self.features.children())[:-5])
        self.features_conv5_2 = nn.Sequential(*list(self.features.children())[-5:-3])
        self.features_conv5_3 = nn.Sequential(*list(self.features.children())[-3:-1])
        self.bilinear_proj = nn.Sequential(nn.Conv2d(512, 8192, kernel_size=1, bias=False),
                                           nn.BatchNorm2d(8192),
                                           nn.ReLU(inplace=True))
        # Linear classifier.
        self.fc = torch.nn.Linear(8192 * 3, num_classes)

    def hbp(self, conv1, conv2):
        n = conv1.size()[0]
        proj_1 = self.bilinear_proj(conv1)
        proj_2 = self.bilinear_proj(conv2)
        assert (proj_1.size() == (n, 8192, 28, 28))
        x = proj_1 * proj_2
        assert (x.size() == (n, 8192, 28, 28))
        x = torch.sum(x.view(x.size()[0], x.size()[1], -1), dim=2)
        x = x.view(n, 8192)
        x = torch.sqrt(x + 1e-5)
        x = torch.nn.functional.normalize(x)
        return x

    def forward(self, x):
        n = x.size()[0]
        assert x.size() == (n, 3, 448, 448)
        x_conv5_1 = self.features_conv5_1(x)
        x_conv5_2 = self.features_conv5_2(x_conv5_1)
        x_conv5_3 = self.features_conv5_3(x_conv5_2)

        x_branch_1 = self.hbp(x_conv5_1, x_conv5_2)
        x_branch_2 = self.hbp(x_conv5_2, x_conv5_3)
        x_branch_3 = self.hbp(x_conv5_1, x_conv5_3)

        x_branch = torch.cat([x_branch_1, x_branch_2, x_branch_3], dim=1)
        assert x_branch.size() == (n, 8192 * 3)

        x = self.fc(x_branch)
        return x
