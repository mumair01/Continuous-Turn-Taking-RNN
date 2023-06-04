# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2023-05-26 16:20:30
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-06-04 11:59:21

import pytest

from turn_taking.dsets.maptask.maptask import MapTaskDataReader


@pytest.mark.data
# @pytest.mark.parametrize("variant", ["prosody", "full"])
def test_maptask_data_reader(save_dir, force_reprocess):
    maptask = MapTaskDataReader(num_conversations=6)
    maptask.prepare_data()
    # Test will make sure that there are no exceptions thrown.
    maptask.setup(save_dir=save_dir, force_reset=False)
    for k, v in maptask.data_paths.items():
        print(k, v)
