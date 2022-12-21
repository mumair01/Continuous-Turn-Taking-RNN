# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-12-20 13:17:53
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-21 12:02:01

import sys
import os
import pytest


from src.dm import MapTaskVADDM
from data_lib.maptask import MapTaskDataReader

def test_maptask_va_dm():
    reader = MapTaskDataReader(
        num_conversations=2
    )
    reader.prepare_data()
    reader.setup(
        variant="full",
        save_dir="./output"
    )