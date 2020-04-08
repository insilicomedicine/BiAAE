from torch import nn

class FCDiscriminator(nn.Module):
    def __init__(self, in_dim, use_sigmoid=False):
        super(FCDiscriminator, self).__init__()
        self.fc = self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 1),
        )
        
        if use_sigmoid:
            self.fc.add_module('sigmoid', nn.Sigmoid())

    def forward(self, x):
        return self.fc(x)
