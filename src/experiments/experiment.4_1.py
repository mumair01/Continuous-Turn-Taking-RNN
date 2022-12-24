# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-12-22 13:34:59
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-24 09:23:59

from hydra_zen import just, instantiate, to_yaml

from data_lib.maptask.dm import MapTaskPauseDataModule, MapTaskVADataModule
from models.va_predictor import VoiceActivityPredictors

import pytorch_lightning as pl

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

