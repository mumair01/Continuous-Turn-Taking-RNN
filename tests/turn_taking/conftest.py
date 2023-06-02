# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2023-05-31 10:41:29
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-05-31 11:10:43


import pytest
import sys
import os
from dataclasses import dataclass


_ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
_RESULTS_ROOT = os.path.join(_ROOT_PATH, "test_results", "turn_taking")


@pytest.fixture
def result_dir():
    return _RESULTS_ROOT
