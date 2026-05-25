import torch 
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self,dim:int,p_drop:float=0.1):
        super().__init__()
        self.line1 = nn.Linear(dim,dim)
        self.line2 = nn.Linear(dim,dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p_drop)
        
    def forward(self,x):
        h = self.act(self.line1(x))
        h = self.drop(h)
        h = self.line2(h)
        
        return self.act(x + h)
        

class NEIMSModel(nn.Module):
    def __init__(self,n_bits=4096, width=2000, depth=7, max_mz=2000, p_drop=0.1, tau=1):
        super().__init__()
        self.L = max_mz + 1
        self.tau = int(tau)

        layers = [nn.Linear(n_bits, width), nn.ReLU(), nn.Dropout(p_drop)]
        for _ in range(depth - 1):
            layers.append(ResidualBlock(width, p_drop))
        self.trunk = nn.Sequential(*layers)

        self.head_forward = nn.Linear(width, self.L)
        self.head_reverse = nn.Linear(width, self.L)
        self.head_gate = nn.Linear(width, self.L)
        self.softplus = nn.Softplus()
        
    
    def _reverse_reindex(self, pr_base: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        B, L = pr_base.shape
        device = pr_base.device
        idx_i = torch.arange(L, device=device).view(1, -1).expand(B, -1)  # [B, L]
        shift = (masses + self.tau).view(-1, 1)                             # [B, 1]
        idx_j = shift - idx_i                                               # [B, L]
        valid = (idx_j >= 0) & (idx_j < L)
        idx_jc = torch.clamp(idx_j, 0, L - 1).long()

        out = torch.zeros_like(pr_base)
        out[valid] = pr_base.gather(1, idx_jc)[valid]
        return out

    def forward(self, x: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        feats = self.trunk(x)
        pf = self.head_forward(feats)
        pr_base = self.head_reverse(feats)
        gate = torch.sigmoid(self.head_gate(feats))
        pr = self._reverse_reindex(pr_base, masses)
        out = gate * pf + (1.0 - gate) * pr
        out = self.softplus(out)

        # physics mask: zero out bins > mass + tau
        B, L = out.shape
        device = out.device
        idx = torch.arange(L, device=device).view(1, -1).expand(B, -1)
        cutoff = (masses + self.tau).view(-1, 1)
        mask = idx > cutoff
        out = out.masked_fill(mask, 0.0)
        return out
    
    
class WeightedMSELoss(nn.Module):
    def __init__(self, max_mz: int=2000):
        super().__init__()
        mz = torch.arange(max_mz + 1, dtype=torch.float32)
        mz[0] = 1.0
        self.register_buffer("weights", torch.sqrt(mz))

    def forward(self, pred: torch.Tensor, target: torch.Tensor, masses: torch.Tensor):
        B, L = pred.shape
        w = self.weights[:L].unsqueeze(0).expand(B, -1)
        idx = torch.arange(L, device=pred.device).unsqueeze(0).expand(B, -1)
        mask = idx <= masses.view(-1, 1)
        se = (pred - target) ** 2
        num = (se * w * mask.float()).sum(dim=1)
        den = (w * mask.float()).sum(dim=1) + 1e-8
        return (num / den).mean()


class Metric:
    @staticmethod
    @torch.no_grad()
    def weighted_cosine(a, b) -> float:
        if isinstance(a,torch.Tensor):
            a = a.cpu().numpy()
        if isinstance(b,torch.Tensor):
            b = b.cpu().numpy()
        L = a.shape[0]
        mz = np.arange(L, dtype=np.float32)
        # mz[0] = 1.0
        w = np.sqrt(mz)
        aw, bw = a * w, b * w
        num = np.dot(aw, bw)
        den = np.linalg.norm(aw) * np.linalg.norm(bw) + 1e-8
        return float((num / den))