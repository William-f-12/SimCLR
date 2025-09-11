import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

def build_resnet(arch="resnet18", pretrained=False, cifar_head=False):
    if arch == "resnet18":
        net = tvm.resnet18(weights=None if not pretrained else tvm.ResNet18_Weights.DEFAULT)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
    else:
        raise NotImplementedError(arch)
    
    if cifar_head:
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()

    return net, feat_dim



class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=1) # p = 2 by default