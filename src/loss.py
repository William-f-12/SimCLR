import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        z1, z2: [B, D]  already L2 normalized
        return: scalar loss
        """
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)

        sim = (z @ z.T) / self.temperature
        sim.fill_diagonal_(float('-inf')) # mask out self-contrast cases

        targets = torch.arange(batch_size, device=z.device)
        targets = torch.cat([targets + batch_size, targets], dim=0)  # 1st B -> 2nd B，2nd B -> 1st B

        # do cross entropy on fp32
        loss = F.cross_entropy(sim.to(dtype=torch.float32), targets)
        return loss
    