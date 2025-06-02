import fire
import json
import torch
from data import SyntheticCLGrandStaffDataset
from smt_trainer import SMT_Trainer

from ExperimentConfig import experiment_config_from_dict
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

torch.set_float32_matmul_precision('high')

def main(config_path, starting_checkpoint):

    with open(config_path, "r") as f:
        config = experiment_config_from_dict(json.load(f))

    datamodule = SyntheticCLGrandStaffDataset(config=config.data)

    Th, Tw = datamodule.train_set.get_max_hw()
    Tl = datamodule.train_set.get_max_seqlen()

    vh, vw = datamodule.val_set.get_max_hw()
    vl = datamodule.val_set.get_max_seqlen()

    th, tw = datamodule.test_set.get_max_hw()
    tl = datamodule.test_set.get_max_seqlen()

    max_height = max(Th, vh, th)
    max_width = max(Tw, vw, tw)
    max_len = max(Tl, vl, tl)

    model_wrapper = SMT_Trainer.load_from_checkpoint(starting_checkpoint, maxh=int(max_height), maxw=int(max_width), maxlen=int(max_len),
                                out_categories=len(datamodule.train_set.w2i), padding_token=datamodule.train_set.w2i["<pad>"],
                                in_channels=1, w2i=datamodule.train_set.w2i, i2w=datamodule.train_set.i2w,
                                d_model=256, dim_ff=256, num_dec_layers=8)
    model_wrapper.set_stage(datamodule.train_set.curriculum_stage_beginning)
    model_wrapper.set_stage_calculator(datamodule.train_set.get_stage_calculator())

    wandb_logger = WandbLogger(project='SMT-FP', group="GrandStaff", name="SMT-FP-CL", log_model=False)

    early_stopping = EarlyStopping(monitor="val_SER", min_delta=0.01, patience=5, mode="min", verbose=True)

    checkpointer = ModelCheckpoint(dirpath=config.checkpoint.dirpath, filename=config.checkpoint.filename,
                                   monitor=config.checkpoint.monitor, mode=config.checkpoint.mode,
                                   save_top_k=config.checkpoint.save_top_k, verbose=config.checkpoint.verbose,
                                   enable_version_counter=False)
    stage_checkpointer = ModelCheckpoint(dirpath=config.checkpoint.dirpath, filename="SMT-CL-stage",
                                   monitor="stage_step", mode="max",
                                   save_top_k=datamodule.train_set.num_cl_steps, verbose=True)

    trainer = Trainer(max_epochs=10000, min_steps=300000,
                      check_val_every_n_epoch=5,
                      logger=wandb_logger, callbacks=[checkpointer, stage_checkpointer, early_stopping], precision='16-mixed')
    datamodule.train_set.set_trainer_data(trainer)

    trainer.fit(model_wrapper,datamodule=datamodule)

    model = SMT_Trainer.load_from_checkpoint(checkpointer.best_model_path)

    trainer.test(model, datamodule=datamodule)

def launch(config_path, starting_checkpoint):
    main(config_path, starting_checkpoint)

if __name__ == "__main__":
    fire.Fire(launch)
