# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-12-22 13:34:59
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-01-02 16:49:54

import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Hydra imports
from hydra.core.config_store import ConfigStore
from hydra_zen import (
    make_config, instantiate, MISSING, launch,
)

# Pytorch imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger

# Local imports
from data_lib.maptask import  MapTaskVADataModule
from experiments.config import MaptaskVADMConf, TrainerConf
from utils import  get_output_dir

# Logger
import logging
logger = logging.getLogger(__name__)


############
# GLOBALS
############

EXPERIMENT_NAME = "experiment_4_1"

SEQUENCE_LENGTHS_MS = [60_000]
PREDICTION_LENGTHS_MS = [250, 500, 1000, 2000, 3000]
TARGET_PARTICIPANTS = ["f", "g"]

# SEQUENCE_LENGTHS_MS = [60_000]
# PREDICTION_LENGTHS_MS = [250]
# TARGET_PARTICIPANTS = ["f"]

############
# Experiment Configurations
############

# NOTE This is replacing config.yaml from regular hydra.
ExperimentConfig = make_config(
    defaults=[
        "_self_",
        {"model" : None},
    ],
    dm=MaptaskVADMConf,
    model=MISSING,
    trainer=TrainerConf,
    seed=1,
    # TODO: Add more experiment constants here if needed.
)
# Store here so that values can be changed from the command line.
cs = ConfigStore.instance()
cs.store(name="config", node=ExperimentConfig)



############
# Experiment Task Function Definition
############

def task(cfg : ExperimentConfig):
    """
    In this experiment, we want to generate the prediction performance
    for the VAPredictor using the VADataModule for various sequence lengths,
    prediction lengths, and feature sets.
    """
    logger.info(f"Starting experiment 4.1 with configurations:\n{cfg}")

    # Initialize the data module - which are all builds that assume that
    # all MISSING fields have been passed to cfg.
    logger.info("Preparing data module")
    dm : MapTaskVADataModule = instantiate(cfg.dm)
    dm.prepare_data()
    dm.setup(stage="fit")

    # Instantiate the model itself - which are pbuilds since the input dims
    # have to be inferred from the dm.
    logger.info("Preparing model")
    model = instantiate(cfg.model)(
        input_dim=next(iter(dm.train_dataloader()))[0].shape[-1],
        #  NOTE: 10 is hard coded for now as the frame step size
        out_features=int(cfg.dm.prediction_length_ms/10)
    )

    ##### Instantiate the trainer components

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(os.getcwd(),"checkpoints"),
        save_top_k=1,
        monitor="val_loss"
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", mode="min"
    )

    # Logger
    # NOTE: Since mlflow and hydra multi-run do not interact well,
    # need to configure this per experiment.
    mlf_logger = MLFlowLogger(
        experiment_name=EXPERIMENT_NAME,
        run_name=(
            f"seq_length_ms={cfg.dm.sequence_length_ms}."
            f"prediction_length_ms={cfg.dm.prediction_length_ms}."
            f"s0={cfg.dm.target_participant}."
            f"feature_set={dm.feature_set}"
        ),
        tracking_uri=f"file:{get_output_dir()}/{EXPERIMENT_NAME}/ml-runs"
    )

    # Trainer
    logger.info("Preparing trainer")
    trainer = instantiate(cfg.trainer)(
        callbacks=[
            checkpoint_callback, early_stopping_callback
        ],
        logger=mlf_logger
    )

    #### Training
    # Fit and evaluate best model.
    logger.info("Starting training...")
    trainer.fit(model, datamodule=dm)
    trainer.test(ckpt_path=checkpoint_callback.best_model_path,datamodule=dm)
    logger.info("Completed experiment")

def main():

    # Create the sweep directory path
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H-%M-%S")

    # Try all combinations on the variants individually
    for feature_set in ("full", "prosody"):
        sweep_path = f"{get_output_dir()}/{EXPERIMENT_NAME}/{feature_set}/{date}/{time}"
        # Define feature set specific params.
        model = "FullModel" if feature_set == "full" else "ProsodyModel"

        # TODO: Remove the first job after testing.
        (jobs,) = launch(  # type: ignore
            ExperimentConfig,
            task,
            overrides=[
                f"dm.sequence_length_ms={','.join([str(x) for x in SEQUENCE_LENGTHS_MS])}",
                f"dm.prediction_length_ms={','.join([str(x) for x in PREDICTION_LENGTHS_MS])}",
                f"dm.target_participant={','.join(TARGET_PARTICIPANTS)}",
                f"dm.feature_set={feature_set}",
                f"model={model}",
                f"hydra.sweep.dir={sweep_path}"
            ],
            multirun=True,
        )

if __name__ == "__main__":
    main()

