# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-12-24 20:55:30
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-25 14:37:17

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

from utils import (
    get_cache_data_dir, get_output_dir, get_root_path
)


###############
# Custom Builds
###############
sbuilds = make_custom_builds_fn(populate_full_signature=True)
pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)


###############
# DataModules
###############


# Configuring fully with default values, which can and should be overwritten later.
MaptaskVADMConf = builds(
    MapTaskVADataModule,
    data_dir=os.path.join(get_cache_data_dir(),"maptask"),
    sequence_length_ms=MISSING,
    prediction_length_ms=MISSING,
    feature_set=MISSING,
    target_participant=MISSING,
    batch_size=1,
)

MaptaskPauseDMConf = builds(
    MapTaskPauseDataModule,
    data_dir=os.path.join(get_cache_data_dir(),"maptask"),
    sequence_length_ms=MISSING,
    min_pause_length_ms=MISSING,
    max_future_silence_window_ms=MISSING,
    target_participant=MISSING,
    feature_set=MISSING,
    batch_size=1
)


###############
# Models
###############

FullModelConf = pbuilds(
    VoiceActivityPredictor,
    input_dim=130, # This is based on the no. of features of the full dataset.
    hidden_dim=40, # Number of features of the hidden state
    out_features=MISSING, # int(prediction_length_ms/frame_step_size)
    num_layers=1
)

# TODO: Verify dims.
ProsodyModelConf = pbuilds(
    VoiceActivityPredictor,
    input_dim=10, # This is based on the no. of features of the full dataset.
    hidden_dim=40, # Number of features of the hidden state
    out_features=MISSING, # int(prediction_length_ms/frame_step_size)
    num_layers=1
)


###############
# Lightning Trainers
###############

TrainerConf = pbuilds(
    pl.Trainer,
    log_every_n_steps=2,
    max_epochs=1

)

cs = ConfigStore.instance()
cs.store(group="dm", name="maptaskPause", node=MaptaskPauseDMConf)
cs.store(group="dm", name="maptaskVA", node=MaptaskVADMConf)
cs.store(group="model", name="FullModel", node=FullModelConf)
cs.store(group="trainer", name="trainer", node=TrainerConf)
