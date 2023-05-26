# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2023-01-01 11:15:38
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-05-22 10:48:30


import numpy as np
from turn_taking.data_lib.utils import (
    access_dataset,
    access_group,
    create_dataset,
)

H5_FILEPATH = "./h5_test.hdf5"


def test_create_dataset():
    group = access_group(H5_FILEPATH, "root")
    subgroup = access_group(H5_FILEPATH, "root/subgroup")
    subgroup = access_group(H5_FILEPATH, "root/subgroup")
    assert subgroup == subgroup


def test_create_dataset():
    group = access_group(H5_FILEPATH, "root")
    dset = create_dataset(
        H5_FILEPATH,
        "root",
        "test",
        data=np.zeros((1000, 10000)),
        maxshape=(None, None),
    )
    print(dset)
