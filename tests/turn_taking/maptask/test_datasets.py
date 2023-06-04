# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2023-05-31 11:38:43
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-06-04 12:40:07


import pytest
from turn_taking.dsets.maptask.datasets import (
    MapTaskPauseDataset,
    MapTaskVADDataset,
)


@pytest.mark.data
@pytest.mark.parametrize("sequence_length_ms", [10_000, 5000])
@pytest.mark.parametrize("prediction_length_ms", [500, 1000])
@pytest.mark.parametrize("target_participant", ["f", "g"])
@pytest.mark.parametrize("feature_set", ["full", "prosody"])
def test_maptask_va_dataset(
    sequence_length_ms,
    prediction_length_ms,
    target_participant,
    feature_set,
    save_dir,
    force_reprocess,
):
    print(save_dir)
    dset = MapTaskVADDataset(
        data_dir=save_dir,
        sequence_length_ms=sequence_length_ms,
        prediction_length_ms=prediction_length_ms,
        target_participant=target_participant,
        feature_set=feature_set,
        force_reprocess=force_reprocess,
        num_conversations=4,
    )
    # TODO: Add assert statements for length.
    print(len(dset))


# @pytest.mark.data
# @pytest.mark.parametrize("sequence_length_ms", [10_000, 5000])
# @pytest.mark.parametrize("min_pause_length_ms", [250, 500])
# @pytest.mark.parametrize("max_future_silence_window_ms", [500, 1000])
# @pytest.mark.parametrize("s0_participant", ["f", "g"])
# @pytest.mark.parametrize("feature_set", ["full", "prosody"])
# def test_maptask_pause_dataset(
#     sequence_length_ms,
#     min_pause_length_ms,
#     max_future_silence_window_ms,
#     s0_participant,
#     feature_set,
#     save_dir,
#     force_reprocess,
# ):
#     MapTaskPauseDataset(
#         data_dir=save_dir,
#         sequence_length_ms=sequence_length_ms,
#         min_pause_length_ms=min_pause_length_ms,
#         max_future_silence_window_ms=max_future_silence_window_ms,
#         s0_participant=s0_participant,
#         feature_set=feature_set,
#         force_reprocess=force_reprocess,
#         save_as_csv=True,
#         num_conversations=4,
#     )
