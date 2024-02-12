import os
import gin
import fire
import torch
import torch.nn as nn

from loguru import logger
from data import load_ctc_data, batch_preparation_ctc
from torch.utils.data import DataLoader
from ModelManager import get_model, LighntingE2EModelUnfolding
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

    train_dataset, val_dataset, test_dataset = load_ctc_data(data_path)

    w2i, i2w = train_dataset.get_dictionaries()

    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=20, collate_fn=batch_preparation_ctc, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=20, collate_fn=batch_preparation_ctc)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=20, collate_fn=batch_preparation_ctc)

    max_height, max_width = train_dataset.get_max_hw()
    max_len = train_dataset.get_max_seqlen()

    model, model_torch = get_model(maxwidth=max_width, maxheight=max_height, in_channels=1, 
                      out_size=len(i2w)+1, blank_idx=len(i2w), model_name="CRNN", output_path="out", i2w=i2w)
    
    wandb_logger = WandbLogger(project='ICDAR 2024', group=f"{corpus_name}", name=f"{model_name}", log_model=False)

    early_stopping = EarlyStopping(monitor=metric_to_watch, min_delta=0.01, patience=5, mode="min", verbose=True)
    
    checkpointer = ModelCheckpoint(dirpath=f"weights/{corpus_name}/", filename=f"{model_name}", 
                                   monitor=metric_to_watch, mode='min',
                                   save_top_k=1, verbose=True)

    trainer = Trainer(max_epochs=20, check_val_every_n_epoch=5, logger=wandb_logger, callbacks=[checkpointer, early_stopping])
    
    trainer.fit(model, train_dataloader, val_dataloader)

    model = LighntingE2EModelUnfolding.load_from_checkpoint(checkpointer.best_model_path, model=model_torch)

    trainer.test(model, test_dataloader)

def launch(config):
    gin.parse_config_file(config)
    main()

if __name__ == "__main__":
    fire.Fire(launch)