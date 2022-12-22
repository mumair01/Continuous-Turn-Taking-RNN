# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-12-20 13:17:53
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-22 10:09:22

import sys
import os
import pytest


from data_lib.maptask.dm import MapTaskVADataModule
from data_lib.maptask.maptask import MapTaskDataReader

def test_maptask_reader():
    reader = MapTaskDataReader(
        num_conversations=3
    )
    reader.prepare_data()
    reader.setup(
        variant="full",
        save_dir="./output"
    )
    reader.setup(
        variant="prosody",
        save_dir="./output"
    )

def test_maptask_va_dm():
    dm = MapTaskVADataModule(
        sequence_length_ms=60_000,
        prediction_length_ms=3000,
        target_participant="f",
        frame_step_size_ms=10
    )
    dm.prepare_data(
        # data_dir="/Users/muhammadumair/Documents/Repositories/mumair01-repos/TRP-Detection/output/full",
        save_dir="./output",
        variant="full"
    )
    dm.setup()
    loader = dm.train_dataloader()
    x, y = next(iter(loader))
