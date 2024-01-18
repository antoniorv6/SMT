import os
import gin
import fire
import torch
import torch.nn as nn

from loguru import logger
from data import load_grandstaff_singleSys, batch_preparation_img2seq
from torch.utils.data import DataLoader
from ModelManager import get_DAN_network, Poliphony_DAN
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

torch.set_float32_matmul_precision('high')

@gin.configurable
def main(data_path, corpus_name=None, model_name=None, metric_to_watch=None):
    logger.info("-----------------------")
    logger.info(f"Training with the {model_name} model")
    logger.info("-----------------------")

    data_path = f"{data_path}"
    out_dir = f"out/{corpus_name}/{model_name}"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/hyp", exist_ok=True)
    os.makedirs(f"{out_dir}/gt", exist_ok=True)

    train_dataset, val_dataset, test_dataset = load_grandstaff_singleSys(data_path)

    w2i, i2w = train_dataset.get_dictionaries()

    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=20, collate_fn=batch_preparation_img2seq, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=20, collate_fn=batch_preparation_img2seq)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=20, collate_fn=batch_preparation_img2seq)

    max_height, max_width = train_dataset.get_max_hw()
    max_len = train_dataset.get_max_seqlen()

    model = get_DAN_network(in_channels=1,
                            max_height=max_height, max_width=max_width, 
                            max_len=max_len, 
                            out_categories=len(train_dataset.get_i2w()), w2i=w2i, i2w=i2w, model_name=model_name, out_dir=out_dir)
    
    wandb_logger = WandbLogger(project='ICDAR 2024', group=f"{corpus_name}", name=f"{model_name}", log_model=False)

    early_stopping = EarlyStopping(monitor=metric_to_watch, min_delta=0.01, patience=5, mode="min", verbose=True)
    
    checkpointer = ModelCheckpoint(dirpath=f"weights/{corpus_name}/", filename=f"{model_name}", 
                                   monitor=metric_to_watch, mode='min',
                                   save_top_k=1, verbose=True)

    trainer = Trainer(max_epochs=5000, check_val_every_n_epoch=2, logger=wandb_logger, callbacks=[checkpointer, early_stopping])
    
    trainer.fit(model, train_dataloader, val_dataloader)

    model = Poliphony_DAN.load_from_checkpoint(checkpointer.best_model_path)

    trainer.test(model, test_dataloader)

def launch(config):
    gin.parse_config_file(config)
    main()

if __name__ == "__main__":
    fire.Fire(launch)