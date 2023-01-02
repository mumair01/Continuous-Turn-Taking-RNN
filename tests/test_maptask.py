# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-12-20 13:17:53
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-01-02 12:31:49

import sys
import os
import pytest


from data_lib.maptask.dm import MapTaskVADataModule, MapTaskPauseDataModule
from data_lib.maptask.maptask import MapTaskDataReader
from data_lib.maptask.dsets import MapTask, MapTaskVADDataset, MapTaskPauseDataset
from utils import (
    get_cache_data_dir,
    get_output_dir,
    get_raw_data_dir,
    get_processed_data_dir,
)

MAPTASK_CACHE_DIR = os.path.join(get_cache_data_dir(), "maptask")
MAPTASK_RAW_DIR = os.path.join(get_raw_data_dir(), "maptask")
MAPTASK_TEMP_SAVE_DIR = os.path.join("tests","output","maptask")
FORCE_REPROCESS = True


########
# Testing the MapTaskReader Class
########


@pytest.mark.parametrize(
    "variant", ["prosody", "full"]
)
def test_maptask_data_reader(variant):
    maptask = MapTaskDataReader(
        num_conversations=2
    )
    maptask.prepare_data()
    maptask.setup(
        variant=variant,
        save_dir=MAPTASK_TEMP_SAVE_DIR,
        reset=FORCE_REPROCESS
    )
    print(maptask.data_paths)


def test_maptask_dataset():
    dset = MapTask(
        data_dir=MAPTASK_TEMP_SAVE_DIR,
        num_proc=4
    )

# @pytest.mark.parametrize("sequence_length_ms", [10_000])
# @pytest.mark.parametrize("prediction_length_ms", [500])
# @pytest.mark.parametrize("target_participant", ["f"])
# @pytest.mark.parametrize("feature_set", ["prosody"])


@pytest.mark.parametrize("sequence_length_ms", [10_000, 5000])
@pytest.mark.parametrize("prediction_length_ms", [500, 1000])
@pytest.mark.parametrize("target_participant", ["f", "g"])
@pytest.mark.parametrize("feature_set", ["full", "prosody"])
def test_maptask_va_dataset(
    sequence_length_ms,
    prediction_length_ms,
    target_participant,
    feature_set
):

    dset = MapTaskVADDataset(
        data_dir=MAPTASK_TEMP_SAVE_DIR,
        sequence_length_ms=sequence_length_ms,
        prediction_length_ms=prediction_length_ms,
        target_participant=target_participant,
        feature_set=feature_set,
        force_reprocess=FORCE_REPROCESS,
    )
    print(len(dset))


@pytest.mark.parametrize("sequence_length_ms", [10_000, 5000])
@pytest.mark.parametrize("min_pause_length_ms", [250, 500])
@pytest.mark.parametrize("max_future_silence_window_ms", [500, 1000])
@pytest.mark.parametrize("s0_participant", ["f", "g"])
@pytest.mark.parametrize("feature_set", ["full", "prosody"])
def test_maptask_pause_dataset(
    sequence_length_ms,
    min_pause_length_ms,
    max_future_silence_window_ms,
    s0_participant,
    feature_set
):
    MapTaskPauseDataset(
        data_dir=MAPTASK_TEMP_SAVE_DIR,
        sequence_length_ms=sequence_length_ms,
        min_pause_length_ms=min_pause_length_ms,
        max_future_silence_window_ms=max_future_silence_window_ms,
        s0_participant=s0_participant,
        feature_set=feature_set,
        force_reprocess=FORCE_REPROCESS,
        save_as_csv=True
    )


# def test_maptask_va_dm():
#     dm = MapTaskVADataModule(
#         data_dir="/Users/muhammadumair/Documents/Repositories/mumair01-repos/TRP-Detection/data/cache/maptask",
#         sequence_length_ms=60_000,
#         prediction_length_ms=3000,
#         feature_set="full",
#         target_participant="f",
#         frame_step_size_ms=10,
#         batch_size=1
#     )
#     dm.prepare_data()
#     dm.setup()
#     loader = dm.train_dataloader()
#     x, y = next(iter(loader))
#     print(x.shape, y.shape)

# def test_maptask_pause_dm():
#     dm = MapTaskPauseDataModule(
#         data_dir="/Users/muhammadumair/Documents/Repositories/mumair01-repos/TRP-Detection/data/cache/maptask",
#         sequence_length_ms=60_000,
#         min_pause_length_ms=500,
#         max_future_silence_window_ms=1000,
#         target_participant="f",
#         feature_set="full",
#         batch_size=1
#     )
#     dm.prepare_data()
#     dm.setup()
#     loader = dm.train_dataloader()
#     x, y = next(iter(loader))
#     print(x.shape, y.shape)
