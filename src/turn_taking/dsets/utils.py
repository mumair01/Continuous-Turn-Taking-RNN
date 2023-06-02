# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2023-01-01 10:58:12
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-05-31 09:58:45


import shutil
import sys
import os
import h5py
import numpy as np
from typing import Tuple


def reset_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def check_group_exists(
    filepath: str,
    name: str,
) -> bool:
    with h5py.File(filepath, "a") as f:
        return name in f


def access_group(
    filepath: str,
    name: str,
) -> h5py.Group:
    """
    Given a filepath and a list of keys, each representing subgroups,
    create a dataset with the specified size.
    """
    with h5py.File(filepath, "a") as f:
        if not check_group_exists(filepath, name):
            return f.create_group(name)
        else:
            return f[name]


def create_dataset(
    filepath: str,
    group: str,
    name: str,
    shape: Tuple = None,
    dtype: str = None,
    data: np.ndarray = None,
    **kwargs,
) -> None:
    with h5py.File(filepath, "a") as f:
        group = f.require_group(group)
        if not name in group:
            dset = group.create_dataset(name, shape, dtype, data, **kwargs)
        else:
            dset = data

    return dset


def access_dataset(
    filepath: str,
    group: str,
    name: str,
) -> h5py.Dataset:
    with h5py.File(filepath, "a") as f:
        group = f.require_group(group)
        if name in group:
            return group[name]
