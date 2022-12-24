# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-12-24 09:00:50
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-24 09:15:51

from hydra.core.config_store import ConfigStore
from hydra_zen import builds, make_config, make_custom_builds_fn

import torch
import pytorch_lightning as pl

from data_lib.maptask.dm import MapTaskPauseDataModule, MapTaskVADataModule
from models.va_predictor import VoiceActivityPredictors

# NOTE: Using hydra_zen, we can partially configure the optimizer here.
# Configure the optimizer

# Configure the LightningDataModule

# Configure the Lightning Module

# Configure the Trainer

# Configure the top-level experiment with the desired task function.