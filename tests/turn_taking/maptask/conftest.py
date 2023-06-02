# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2023-05-31 11:11:50
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-05-31 16:51:35

import pytest

import shutil
import sys
import os


_FORCE_REPROCESS = True
_CLEAR_DIRS_AFTER_TEST = False
_RESET_DIRS_BEFORE_TEST = False


def reset_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)


@pytest.fixture
def cache_dir(result_dir):
    path = os.path.join(result_dir, "maptask", "cache")
    if _RESET_DIRS_BEFORE_TEST:
        reset_dir(path)
    yield path
    if _CLEAR_DIRS_AFTER_TEST:
        shutil.rmtree(path)


@pytest.fixture
def save_dir(result_dir):
    path = os.path.join(result_dir, "maptask", "save")
    if _RESET_DIRS_BEFORE_TEST:
        reset_dir(path)
    yield path
    if _CLEAR_DIRS_AFTER_TEST:
        shutil.rmtree(path)


@pytest.fixture
def force_reprocess():
    return _FORCE_REPROCESS
