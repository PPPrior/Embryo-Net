import torch

from .vgg import vgg16

__all__ = ['BCNN']


class BCNN(torch.nn.Module):
    """Bi-linear-CNN.

    The B-CNN model is illustrated as follows:
    conv1^2 (64) -> pool1 -> conv2^2 (128) -> pool2 -> conv3^3 (256) -> pool3
    -> conv4^3 (512) -> pool4 -> conv5^3 (512) -> bilinear pooling
    -> sqrt-normalize -> L2-normalize -> fc (200).

    The network accepts a 3 x 448 x 448 input, and the pool5 activation has
    shape 512 x 28 x 28 since we down-sample 4 times.

    Attributes:
        features, torch.nn.Module: Convolution and pooling layers.
        fc, torch.nn.Module: 2.
    """

    def __init__(self, num_classes):
        """Declare all needed layers."""
        torch.nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.num_classes = num_classes
        self.features = vgg16(pretrained=False).features
        self.features = torch.nn.Sequential(*list(self.features.children())[:-1])  # Remove pool5.
        # Linear classifier.
        self.fc = torch.nn.Linear(512 ** 2, num_classes)

    def forward(self, x):
        """Forward pass of the network.

        Args:
            x, torch.Tensor of shape N x 3 x 448 x 448.

        Returns:
            Score, torch.Tensor of shape N x 200.
        """
        x = self.features(x)
        x = x.view(-1, 512, 28 ** 2)
        x = torch.bmm(x, torch.transpose(x, 1, 2)) / (28 ** 2)  # Bi-linear
        x = x.view(-1, 512 ** 2)
        x = torch.sqrt(x + 1e-5)
        x = torch.nn.functional.normalize(x)
        x = self.fc(x)
        return x
