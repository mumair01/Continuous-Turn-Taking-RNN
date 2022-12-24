# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-12-20 13:17:53
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-24 14:28:49

import sys
import os
import pytest


from data_lib.maptask.dm import MapTaskVADataModule, MapTaskPauseDataModule
from data_lib.maptask.maptask import MapTaskDataReader
from data_lib.maptask.dsets import MapTask, MapTaskVADDataset, MapTaskPauseDataset

def test_maptask():
    dset = MapTask(
        data_dir="/Users/muhammadumair/Documents/Repositories/mumair01-repos/TRP-Detection/data/cache/maptask",
    )


def test_maptask_va_dataset():
    dset = MapTaskVADDataset(
        data_dir="/Users/muhammadumair/Documents/Repositories/mumair01-repos/TRP-Detection/data/cache/maptask",
        sequence_length_ms=10_000,
        prediction_length_ms=1000,
        target_participant="f",
        feature_set="full",
        force_reprocesses=False
    )
    dset = MapTaskVADDataset(
        data_dir="/Users/muhammadumair/Documents/Repositories/mumair01-repos/TRP-Detection/data/cache/maptask",
        sequence_length_ms=10_000,
        prediction_length_ms=1000,
        target_participant="f",
        feature_set="prosody",
        force_reprocesses=False
    )

def test_maptask_pause_dataset():
    MapTaskPauseDataset(
        data_dir="/Users/muhammadumair/Documents/Repositories/mumair01-repos/TRP-Detection/data/cache/maptask",
        sequence_length_ms=60_000,
        min_pause_length_ms=500,
        max_future_silence_window_ms=1000,
        s0_participant="f",
        feature_set="full",
        force_reprocess=False
    )
    MapTaskPauseDataset(
        data_dir="/Users/muhammadumair/Documents/Repositories/mumair01-repos/TRP-Detection/data/cache/maptask",
        sequence_length_ms=60_000,
        min_pause_length_ms=500,
        max_future_silence_window_ms=1000,
        s0_participant="f",
        feature_set="prosody",
        force_reprocess=False
    )

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
        data_dir="/Users/muhammadumair/Documents/Repositories/mumair01-repos/TRP-Detection/data/cache/maptask",
        sequence_length_ms=60_000,
        prediction_length_ms=3000,
        feature_set="full",
        target_participant="f",
        frame_step_size_ms=10,
        batch_size=1
    )
    dm.prepare_data()
    dm.setup()
    loader = dm.train_dataloader()
    x, y = next(iter(loader))
    print(x.shape, y.shape)

def test_maptask_pause_dm():
    dm = MapTaskPauseDataModule(
        data_dir="/Users/muhammadumair/Documents/Repositories/mumair01-repos/TRP-Detection/data/cache/maptask",
        sequence_length_ms=60_000,
        min_pause_length_ms=500,
        max_future_silence_window_ms=1000,
        target_participant="f",
        feature_set="full",
        batch_size=1
    )
    dm.prepare_data()
    dm.setup()
    loader = dm.train_dataloader()
    x, y = next(iter(loader))
    print(x.shape, y.shape)
