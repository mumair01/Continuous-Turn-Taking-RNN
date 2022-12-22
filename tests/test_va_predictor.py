# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-12-22 11:59:45
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-22 13:43:33

import pytest
import sys
import os

from models.va_predictor import VoiceActivityPredictor
from data_lib.maptask.dm import MapTaskVADataModule

from pytorch_lightning import Trainer

def test_va_predictor():
    model = VoiceActivityPredictor(
        input_dim=130,
        hidden_dim=40, # Number of features of the hidden state
        out_features=30, # Number of output time steps,
        num_layers=1
    )

    dm = MapTaskVADataModule(
        sequence_length_ms=10_000,
        prediction_length_ms=1000,
        frame_step_size_ms=10,
        data_dir="/Users/muhammadumair/Documents/Repositories/mumair01-repos/TRP-Detection/data/processed/maptask/full",
        target_participant="f",
        batch_size=1
    )
    dm.prepare_data()
    dm.setup(stage="fit")
    trainer = Trainer(
        max_epochs=1
    )
    trainer.fit(model, dm)

