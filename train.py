import fire
import json
import torch
from data import GrandStaffDataset
from smt_trainer import SMT_Trainer

from ExperimentConfig import experiment_config_from_dict
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

torch.set_float32_matmul_precision('high')

def main(config_path):

    with open(config_path, "r") as f:
        config = experiment_config_from_dict(json.load(f))

    datamodule = GrandStaffDataset(config=config.data)

    max_height = datamodule.get_max_height()
    max_width = datamodule.get_max_width()
    max_len = datamodule.get_max_length()

    model_wrapper = SMT_Trainer(maxh=int(max_height), maxw=int(max_width), maxlen=int(max_len),
                                out_categories=len(datamodule.train_set.w2i), padding_token=datamodule.train_set.w2i["<pad>"],
                                in_channels=1, w2i=datamodule.train_set.w2i, i2w=datamodule.train_set.i2w,
                                d_model=256, dim_ff=256, num_dec_layers=8)

    wandb_logger = WandbLogger(project='SMT_Reimplementation', group="GrandStaff", name="SMT_NexT_GrandStaff", log_model=False)

    early_stopping = EarlyStopping(monitor="val_SER", min_delta=0.01, patience=5, mode="min", verbose=True)

    checkpointer = ModelCheckpoint(dirpath=config.checkpoint.dirpath, filename=config.checkpoint.filename,
                                   monitor=config.checkpoint.monitor, mode=config.checkpoint.mode,
                                   save_top_k=config.checkpoint.save_top_k, verbose=config.checkpoint.verbose)

    trainer = Trainer(max_epochs=10000,
                      check_val_every_n_epoch=5,
                      logger=wandb_logger, callbacks=[checkpointer, early_stopping], precision='16-mixed')

    trainer.fit(model_wrapper,datamodule=datamodule)

    model = SMT_Trainer.load_from_checkpoint(checkpointer.best_model_path)

    trainer.test(model, datamodule=datamodule)

def launch(config_path):
    main(config_path)

if __name__ == "__main__":
    fire.Fire(launch)
