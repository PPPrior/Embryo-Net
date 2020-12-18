from torch import nn

import model

__all__ = ['Fusion']


class ConsensusModule(nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, x):
        self.shape = x.size()
        x = x.view(-1, 4, x.size()[-1])
        if self.consensus_type == 'avg':
            output = x.mean(dim=self.dim)
        elif self.consensus_type == 'max':
            output = x.max(dim=self.dim).values
        else:
            output = None

        return output


class Fusion(nn.Module):
    def __init__(self, base_model, consensus_type, num_classes=2, pretrained=True):
        super(Fusion, self).__init__()
        self.base_model = getattr(model, base_model)(pretrained)
        feature_dim = getattr(self.base_model, 'fc').in_features

        self.consensus = ConsensusModule(consensus_type)
        self.fc = nn.Linear(1000, num_classes)

    def forward(self, xs):
        x = xs.view((-1, 3) + xs.size()[-2:])
        x = self.base_model(x)
        x = self.consensus(x)
        x = self.fc(x)
        return x
