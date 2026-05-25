import lightning as L
import inspect
import torch
import importlib
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
from .model_neims import NEIMSModel,WeightedMSELoss, Metric



class MInterface(L.LightningModule):
    def __init__(self, lr:float=None, **kargs):
        super().__init__()
        # # 把 __init__ 里的参数自动保存到 self.hparams 中，方便后续调用和保存
        # self.save_hyperparameters()
        self.model = NEIMSModel(n_bits=4096, width=2000, depth=7, max_mz=2000, p_drop=0.1, tau=1)
        self.loss = WeightedMSELoss(max_mz=2000)
        self.lr = lr

    def forward(self, x,mass):
        return self.model(x,mass)

    def training_step(self, batch, batch_idx):
        fp, spec, mass, _ = batch
        pred = self(fp, mass)
        loss = self.loss(pred, spec, mass)
        self.log("train/wMSE", loss, prog_bar=True,on_step=True,on_epoch=True)
        return loss
        

    def validation_step(self, batch, batch_idx):
        fp, spec, mass, _ = batch
        pred = self(fp, mass)
        loss = self.loss(pred, spec, mass)
        self.log("val/wMSE", loss, prog_bar=True,  on_step=False,on_epoch=True,sync_dist=True)
        return loss


    def test_step(self, batch, batch_idx):
        fp, spec, mass, _ = batch
        pred = self(fp, mass)
        loss = self.loss(pred, spec, mass)
        # weighted cosine (batch mean)
        cos = sum(Metric.weighted_cosine(pred[i], spec[i]) for i in range(fp.size(0))) / max(1, fp.size(0))
        self.log("test/wMSE", loss, prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        self.log("test/wCos", cos, prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        return {"test_wmse":loss,"test_wcos":cos}

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=weight_decay)

        if self.lr_scheduler is None:
            return optimizer
        else:
            if self.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.lr_decay_steps,
                                       gamma=self.lr_decay_rate)
            elif self.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=100,
                                                  eta_min=self.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1, 
                }
            }