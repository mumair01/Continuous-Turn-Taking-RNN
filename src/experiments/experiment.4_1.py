# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-12-22 13:34:59
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-24 15:46:02

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from hydra_zen import just, instantiate, to_yaml

from data_lib.maptask import MapTaskPauseDataModule, MapTaskVADataModule
from models.va_predictor import VoiceActivityPredictor

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from utils import get_cache_data_dir, get_output_dir, get_root_path


def experiment_4_1(cfg):
    """
    In this experiment, we want to generate the prediction performance
    for the VAPredictor using the VADataModule for various sequence lengths,
    prediction lengths, and feature sets.
    """
    pl.seed_everything(cfg.seed)

    # Create the save dirs

    # Instantiate the LightningDataModule

    # Instantiate the LightningModule

    # Start training

    # Evaluate the models

    # Save the best model, the training data, and the losses.


    obj = instantiate(cfg)

    lit_module = obj.lit_module()

    obj.trainer.fit(lit_module)


if __name__ == "__main__":

    dm = MapTaskVADataModule(
        data_dir=os.path.join(get_cache_data_dir(),"maptask"),
        sequence_length_ms=60_000,
        prediction_length_ms=3000,
        feature_set="full",
        target_participant="f",
        frame_step_size_ms=10,
        batch_size=1
    )
    dm.prepare_data()
    dm.setup(stage="fit")
    # train_loader = dm.train_dataloader()
    # x, y = next(iter(train_loader))
    # print(x.shape, y.shape)

    model = VoiceActivityPredictor(
        input_dim=130,
        hidden_dim=40, # Number of features of the hidden state
        out_features=300, # Number of output time steps,
        num_layers=1
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=get_output_dir(),
        save_top_k=1,
        monitor="val_loss"
    )
    trainer = pl.Trainer(
        callbacks=[
            EarlyStopping(monitor="val_loss",mode="min"),
            checkpoint_callback
        ],
        logger=CSVLogger(
            save_dir=get_output_dir()
        ),
        log_every_n_steps=5,
        max_epochs=1
    )

    trainer.fit(model, datamodule=dm)

    # Eval the best model.
    res = trainer.test(ckpt_path=checkpoint_callback.best_model_path,datamodule=dm)
    print(res)


