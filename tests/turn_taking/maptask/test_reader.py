# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2023-05-26 16:20:30
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-06-04 14:24:42

import pytest

from turn_taking.dsets.maptask.datasets.maptask import MapTaskDataReader


@pytest.mark.data
def test_maptask_data_reader(save_dir, force_reprocess):
    maptask = MapTaskDataReader(num_conversations=2)
    maptask.prepare_data()
    # Test will make sure that there are no exceptions thrown.
    maptask.setup(save_dir=save_dir, force_reset=force_reprocess)
    for k, v in maptask.data_paths.items():
        print(k, v)
