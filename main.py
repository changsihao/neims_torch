import os
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger,CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from data import DInterface
from model import MInterface

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    config = {
        "train_msp": "train.msp",
        "val_msp":   "val.msp",
        "test_msp":  "test.msp",
        "n_bits": 4096,
        "radius": 2,
        "max_mz": 2000,
        "batch_size": 2048,
        "num_workers": 0,
        "epochs": 200,
        "lr": 1e-3,
        "lr_scheduler": "cosine",
        "lr_interval": "epoch",
        "lr_decay_min_lr": 1e-6,
        "weight_decay": 0.0,
        "seed": 1337,
        "accelerator": "gpu",
        "devices": 1,
        "log_dir": "logs",
        "run_name": "neims_run1",
        "default_root_dir": "outputs",
        "precision": "32-true",
    }

    seed_everything(config["seed"], workers=True)

    # ---------- DataModule ----------
    dm = DInterface(
        train_msp=config["train_msp"],
        val_msp=config["val_msp"],
        test_msp=config["test_msp"],
        n_bits=config["n_bits"],
        radius=config["radius"],
        max_mz=config["max_mz"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )
    dm.setup()

    # ---------- LightningModule ----------
    model = MInterface(lr=config["lr"])
    # 手动挂属性以配合 configure_optimizers
    model.weight_decay = config["weight_decay"]
    model.lr_scheduler = config["lr_scheduler"]
    model.lr_interval = config["lr_interval"]
    model.lr_decay_min_lr = config["lr_decay_min_lr"]

    # ---------- Logger & Callbacks ----------
    os.makedirs(config["log_dir"], exist_ok=True)
    tb_logger = TensorBoardLogger(save_dir=config["log_dir"], name=config["run_name"])

    ckpt_cb = ModelCheckpoint(
        monitor="val/wMSE",
        mode="min",
        save_top_k=3,
        filename="epoch{epoch:02d}-valwMSE{val/wMSE:.5f}",
        auto_insert_metric_name=False,
    )
    es_cb = EarlyStopping(
        monitor="val/wMSE",
        mode="min",
        patience=20,
        strict=False,
        verbose=True,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    # ---------- Trainer ----------
    trainer = Trainer(
        max_epochs=config["epochs"],
        accelerator=config["accelerator"],
        devices=config["devices"],
        # strategy="ddp", 
        logger=[tb_logger],
        callbacks=[ckpt_cb, es_cb, lr_cb],
        default_root_dir=config["default_root_dir"],
        log_every_n_steps=10,
        precision=config["precision"],
    )

    # ---------- Fit / Test ----------
    trainer.fit(model, datamodule=dm)
    if config["test_msp"]:
        trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()