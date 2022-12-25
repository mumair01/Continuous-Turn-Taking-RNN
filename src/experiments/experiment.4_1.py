# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-12-22 13:34:59
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-25 15:56:18

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import logging
logger = logging.getLogger(__name__)

from datetime import datetime
now = datetime.now()
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
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
        {"model" : None},
    ],
    dm=MaptaskVADMConf,
    model=MISSING,
    trainer=TrainerConf,
    seed=1,
    # TODO: Add more experiment constants here!
)
# Store here so that values can be changed from the command line.
cs = ConfigStore.instance()
cs.store(name="config", node=ExperimentConfig)


def experiment_4_1(cfg : ExperimentConfig):
    """
    In this experiment, we want to generate the prediction performance
    for the VAPredictor using the VADataModule for various sequence lengths,
    prediction lengths, and feature sets.
    """
    # Instantiate and prepare the data module
    logger.info("Starting experiment 4.1")
    logger.info("Loading data module...")
    dm = instantiate(cfg.dm)
    dm.prepare_data()
    dm.setup(stage="fit")

    # Instantiate model
    logger.info("Initializing model...")
    model = instantiate(cfg.model)(
        out_features= int(cfg.dm.prediction_length_ms/10) # NOTE: 10 is hard coded for now.
    )

    # Define checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(os.getcwd(),"checkpoints"),
        save_top_k=1,
        monitor="val_loss"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", mode="min"
    )

    # Define loggers
    # NOTE: Since mlflow and hydra multi-run do not interact well,
    # need to configure this per experiment.
    mlf_logger = MLFlowLogger(
        experiment_name="experiment_4_1",
        run_name=(
            f"seq_length_ms={cfg.dm.sequence_length_ms}."
            f"prediction_length_ms={cfg.dm.prediction_length_ms}."
            f"s0={cfg.dm.target_participant}."
            f"feature_set={dm.feature_set}"
        ),
        tracking_uri=f"file:{get_output_dir()}/experiment_4.1/ml-runs"
    )
    logger.info("Instantiating trainer...")
    # Define Trainers
    trainer = instantiate(cfg.trainer)(
        callbacks=[
            checkpoint_callback, early_stopping_callback
        ],
        logger=mlf_logger
    )

    # Run training
    logger.info("Fitting model...")
    trainer.fit(model, datamodule=dm)

    # Evaluate the best model.
    res = trainer.test(ckpt_path=checkpoint_callback.best_model_path,datamodule=dm)
    return res

def main():
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H-%M-%S")
    # TODO: Remove the first job after testing.
    (jobs,) = launch(  # type: ignore
        ExperimentConfig,
        experiment_4_1,
        overrides=[
            "dm.sequence_length_ms=60_000",
            "dm.prediction_length_ms=250",
            "dm.target_participant=f",
            "dm.feature_set=full",
            "model=FullModel",
            f"hydra.sweep.dir={get_output_dir()}/experiment_4.1/{date}/{time}"
        ],
        multirun=True,
    )
    # # Full feature set experiment.
    # (jobs,) = launch(  # type: ignore
    #     ExperimentConfig,
    #     experiment_4_1,
    #     overrides=[
    #         "dm.sequence_length_ms=60_000",
    #         "dm.prediction_length_ms=250,500,1000,2000",
    #         "dm.target_participant=f,g",
    #         "dm.feature_set=full",
    #         "model=FullModel"
    #     ],
    #     multirun=True,
    # )
    # # Prosody feature set experiment.
    # (jobs,) = launch(  # type: ignore
    #     ExperimentConfig,
    #     experiment_4_1,
    #     overrides=[
    #         "dm.sequence_length_ms=60_000",
    #         "dm.prediction_length_ms=250,500,1000,2000",
    #         "dm.target_participant=f,g",
    #         "dm.feature_set=prosody",
    #         "model=ProsodyModel"
    #     ],
    #     multirun=True,
    # )


if __name__ == "__main__":
    main()

