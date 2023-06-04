# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2023-05-31 11:31:59
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-06-04 14:26:41

import pytest

from turn_taking.dsets.maptask.dm import (
    MapTaskVADataModule,
    MapTaskPauseDataModule,
)


# TODO: Add assert statements to tests to check for dim.
@pytest.mark.data
@pytest.mark.parametrize("sequence_length_ms", [10_000, 5000])
@pytest.mark.parametrize("prediction_length_ms", [500, 1000])
@pytest.mark.parametrize("target_participant", ["f", "g"])
@pytest.mark.parametrize("feature_set", ["full", "prosody"])
def test_maptask_va_dm(
    sequence_length_ms,
    prediction_length_ms,
    target_participant,
    feature_set,
    save_dir,
    force_reprocess,
):
    dm = MapTaskVADataModule(
        data_dir=save_dir,
        sequence_length_ms=sequence_length_ms,
        prediction_length_ms=prediction_length_ms,
        feature_set=feature_set,
        target_participant=target_participant,
        batch_size=32,
        force_reprocess=False,
    )
    dm.prepare_data()
    dm.setup()
    loader = dm.train_dataloader()
    x, y = next(iter(loader))
    print(x.shape, y.shape)


@pytest.mark.data
@pytest.mark.parametrize("sequence_length_ms", [10_000, 5000])
@pytest.mark.parametrize("min_pause_length_ms", [250, 500])
@pytest.mark.parametrize("max_future_silence_window_ms", [500, 1000])
@pytest.mark.parametrize("s0_participant", ["f", "g"])
@pytest.mark.parametrize("feature_set", ["full", "prosody"])
def test_maptask_pause_dm(
    sequence_length_ms,
    min_pause_length_ms,
    max_future_silence_window_ms,
    s0_participant,
    feature_set,
    save_dir,
    force_reprocess,
):
    dm = MapTaskPauseDataModule(
        data_dir=save_dir,
        sequence_length_ms=sequence_length_ms,
        min_pause_length_ms=min_pause_length_ms,
        max_future_silence_window_ms=max_future_silence_window_ms,
        target_participant=s0_participant,
        feature_set=feature_set,
        batch_size=1,
        force_reprocess=force_reprocess,
    )
    dm.prepare_data()
    dm.setup()
    loader = dm.train_dataloader()
    x, y = next(iter(loader))
    print(x.shape, y.shape)
