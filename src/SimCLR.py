import torch
import torch.nn as nn
from model import build_resnet, ProjectionHead


class SimCLR(nn.Module):
    def __init__(self, arch="resnet18", proj_hidden=2048, proj_out=128, head="cifar"):
        super().__init__()
        self.encoder, feat_dim = build_resnet(arch=arch, cifar_head=(head=="cifar"))
        self.projector = ProjectionHead(in_dim=feat_dim, hidden_dim=proj_hidden, out_dim=proj_out)

    @torch.no_grad()
    def encode(self, x):
        return self.encoder(x)

    def forward(self, x1, x2):
        h1, h2 = self.encoder(x1), self.encoder(x2)
        z1, z2 = self.projector(h1), self.projector(h2)
        return z1, z2