# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2023-05-31 10:41:29
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-06-04 14:04:14


import pytest
import sys
import os
from dataclasses import dataclass


_ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
_RESULTS_ROOT = os.path.join(_ROOT_PATH, "test_results", "turn_taking")
_FORCE_REPROCESS = False


@pytest.fixture
def result_dir():
    return _RESULTS_ROOT


@pytest.fixture
def force_reprocess():
    return _FORCE_REPROCESS
