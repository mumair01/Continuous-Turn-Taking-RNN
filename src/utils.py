# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-12-24 09:24:42
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-24 15:27:22

import sys
import os
import toml
import shutil

PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
PROJECT_CONFIG_PATH = os.path.join(PROJECT_ROOT_DIR, "src/configs/project.toml")

CONFIG_DATA = toml.load(PROJECT_CONFIG_PATH)

def add_root_to_path(func):
    return lambda : os.path.join(PROJECT_ROOT_DIR, func())

def get_root_path():
    """Returns the path to the root of the repo"""
    return PROJECT_ROOT_DIR


def reset_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path,exist_ok=True)

@add_root_to_path
def get_output_dir():
    return CONFIG_DATA["paths"]["outputDir"]

@add_root_to_path
def get_cache_data_dir():
    return CONFIG_DATA["paths"]["datasets"]["cacheDir"]

@add_root_to_path
def get_processed_data_dir():
    return CONFIG_DATA["paths"]["datasets"]["processedDir"]

@add_root_to_path
def get_raw_data_dir():
    return CONFIG_DATA["paths"]["datasets"]["rawDir"]


if __name__ == "__main__":
    print(get_root_path())
    print(get_cache_data_dir())