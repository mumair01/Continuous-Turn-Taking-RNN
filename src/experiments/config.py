# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-12-24 20:55:30
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-01-02 16:56:32

import sys
import os

import hydra
from hydra.core.config_store import ConfigStore
from hydra_zen import (
    builds, make_config, make_custom_builds_fn, instantiate, MISSING
)

from data_lib.maptask import MapTaskPauseDataModule, MapTaskVADataModule
from models.va_predictor import VoiceActivityPredictor

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from utils import (
    get_cache_data_dir, get_output_dir, get_root_path
)

###############
# Paths
###############
MAPTASK_DATA_DIR = os.path.join(get_cache_data_dir(),"maptask")

###############
# Custom Builds
###############
sbuilds = make_custom_builds_fn(populate_full_signature=True)
pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)


###############
# DataModules
###############

# TODO: The batch size should be changed to 1 later.
# Configuring fully with default values, which can and should be overwritten later.
MaptaskVADMConf = builds(
    MapTaskVADataModule,
    data_dir=MAPTASK_DATA_DIR,
    sequence_length_ms=MISSING,
    prediction_length_ms=MISSING,
    feature_set=MISSING,
    target_participant=MISSING,
    batch_size=32,
    train_split=0.8,
    force_reprocess=False
)

MaptaskPauseDMConf = builds(
    MapTaskPauseDataModule,
    data_dir=MAPTASK_DATA_DIR,
    sequence_length_ms=MISSING,
    min_pause_length_ms=MISSING,
    max_future_silence_window_ms=MISSING,
    feature_set=MISSING,
    target_participant=MISSING,
    batch_size=32,
    train_split=0.8,
    force_reprocess=False
)


###############
# Models
###############

FullModelConf = pbuilds(
    VoiceActivityPredictor,
    input_dim=MISSING, # This is based on the no. of features of the full dataset.
    hidden_dim=40, # Number of features of the hidden state
    out_features=MISSING, # int(prediction_length_ms/frame_step_size)
    num_layers=1
)

ProsodyModelConf = pbuilds(
    VoiceActivityPredictor,
    input_dim=MISSING, # This is based on the no. of features of the full dataset.
    hidden_dim=10, # Number of features of the hidden state
    out_features=MISSING, # int(prediction_length_ms/frame_step_size)
    num_layers=1
)

###############
# Lightning Trainer
###############

###### Trainer config
TrainerConf = pbuilds(
    pl.Trainer,
    log_every_n_steps=50,
    max_epochs=1000,
    accelerator="auto",
    auto_select_gpus=True
)

cs = ConfigStore.instance()

cs.store(group="dm", name="maptaskPause", node=MaptaskPauseDMConf)
cs.store(group="dm", name="maptaskVA", node=MaptaskVADMConf)

cs.store(group="model", name="FullModel", node=FullModelConf)
cs.store(group="model", name="ProsodyModel", node=ProsodyModelConf)


cs.store(group="trainer", name="trainer", node=TrainerConf)
