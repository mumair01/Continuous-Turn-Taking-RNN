# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-12-22 13:34:59
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-25 14:39:52

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import hydra
from hydra.core.config_store import ConfigStore
from hydra_zen import (
    builds, make_config, make_custom_builds_fn, instantiate, MISSING, launch,
    to_yaml
)
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger

from data_lib.maptask import MapTaskPauseDataModule, MapTaskVADataModule
from models.va_predictor import VoiceActivityPredictor


from utils import (
    get_cache_data_dir, get_output_dir, get_root_path
)

from experiments.config import(
    MaptaskVADMConf, TrainerConf
)


# This is replacing config.yaml from regular hydra.
ExperimentConfig = make_config(
    defaults=[
        "_self_",
        {"model" : None}
    ],
    dm=MaptaskVADMConf,
    model=MISSING,
    trainer=TrainerConf,
    seed=1
    # TODO: Add more experiment constants here!
)
# Store here so that values can be changed from the command line.
cs = ConfigStore.instance()
cs.store(name="config", node=ExperimentConfig)

print(to_yaml(ExperimentConfig))

def experiment_4_1(cfg : ExperimentConfig):
    """
    In this experiment, we want to generate the prediction performance
    for the VAPredictor using the VADataModule for various sequence lengths,
    prediction lengths, and feature sets.
    """
    # Instantiate and prepare the data module
    dm = instantiate(cfg.dm)
    dm.prepare_data()
    dm.setup(stage="fit")

    # Instantiate model
    model = instantiate(cfg.model)(
        out_features= int(cfg.dm.prediction_length_ms/10) # NOTE: 10 is hard coded for now.
    )

    # Define checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=get_output_dir(),
        save_top_k=1,
        monitor="val_loss"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", mode="min"
    )

    # Define loggers
    logger = MLFlowLogger(
        experiment_name="lightning_logs",
        tracking_uri="file:./ml-runs"
    )

    # Define Trainers
    trainer = instantiate(cfg.trainer)(
        callbacks=[
            checkpoint_callback, early_stopping_callback
        ],
        logger=logger
    )

    # Run training
    trainer.fit(model, datamodule=dm)

    # Evaluate the best model.
    res = trainer.test(ckpt_path=checkpoint_callback.best_model_path,datamodule=dm)
    return res

# @hydra.main(config_path=None, config_name="config")
def main():
    (jobs,) = launch(  # type: ignore
    ExperimentConfig,
    experiment_4_1,

    overrides=[
        "dm.sequence_length_ms=60_000",
        "dm.prediction_length_ms=250,500,1000,2000",
        "dm.target_participant=f,g",
        "dm.feature_set=full",
        "model=FullModel"
    ],
    multirun=True,
)

if __name__ == "__main__":
    main()

