# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-12-24 20:55:30
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-06-05 10:29:20

import sys
import os

import hydra
from hydra.core.config_store import ConfigStore
from hydra.conf import HydraConf, JobConf
from hydra_zen import (
    builds,
    make_config,
    make_custom_builds_fn,
    instantiate,
    MISSING,
    store,
)

import pytorch_lightning as pl
from turn_taking.dsets.maptask import (
    MapTaskPauseDataModule,
    MapTaskVADataModule,
)
from turn_taking.models import GCTTLSTM

from utils import load_user_configs


###############
# Configurable vars and paths.
###############

_USER_CONFIGS = load_user_configs()["config"]


PROJECT_ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
RESULTS_DIR = os.path.join(
    PROJECT_ROOT_DIR, _USER_CONFIGS["paths"]["results_dir_name"]
)
CACHE_DIR = os.path.join(RESULTS_DIR, _USER_CONFIGS["paths"]["cache_dir_name"])

MAPTASK_DATA_DIR = os.path.join(CACHE_DIR, "maptask")


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
    **_USER_CONFIGS["dm"]["shared"],
    **_USER_CONFIGS["dm"]["MaptaskVADM"],
)

MaptaskPauseDMConf = builds(
    MapTaskPauseDataModule,
    data_dir=MAPTASK_DATA_DIR,
    sequence_length_ms=MISSING,
    min_pause_length_ms=MISSING,
    max_future_silence_window_ms=MISSING,
    feature_set=MISSING,
    target_participant=MISSING,
    **_USER_CONFIGS["dm"]["shared"],
    **_USER_CONFIGS["dm"]["MaptaskPauseDM"],
)


###############
# Models
###############

FullModelConf = pbuilds(
    GCTTLSTM,
    input_dim=MISSING,  # This is based on the no. of features of the full dataset.
    # hidden_dim=40,  # Number of features of the hidden state
    out_features=MISSING,  # int(prediction_length_ms/frame_step_size)
    # num_layers=1,
    **_USER_CONFIGS["model"]["full"],
)

ProsodyModelConf = pbuilds(
    GCTTLSTM,
    input_dim=MISSING,  # This is based on the no. of features of the full dataset.
    # hidden_dim=10,  # Number of features of the hidden state
    out_features=MISSING,  # int(prediction_length_ms/frame_step_size)
    # num_layers=1,
    **_USER_CONFIGS["model"]["prosody"],
)

###############
# Lightning Trainer
###############

###### Trainer config
TrainerConf = pbuilds(
    pl.Trainer,
    **_USER_CONFIGS["pl"]["trainer"]
    # log_every_n_steps=1,
    # max_epochs=1000,
    # accelerator="auto",
    # auto_select_gpus=True,
)

cs = ConfigStore.instance()

# NOTE: This allows us tp configure the hydra defaults.
store(HydraConf(**_USER_CONFIGS["hydra_zen"]))
store.add_to_hydra_store()

cs.store(group="dm", name="maptaskPause", node=MaptaskPauseDMConf)
cs.store(group="dm", name="maptaskVA", node=MaptaskVADMConf)

cs.store(group="model", name="FullModel", node=FullModelConf)
cs.store(group="model", name="ProsodyModel", node=ProsodyModelConf)


cs.store(group="trainer", name="trainer", node=TrainerConf)
