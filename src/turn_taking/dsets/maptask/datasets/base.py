# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2023-05-31 11:34:26
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-05-31 16:50:12


import sys
import os
import glob

import pandas as pd
from datasets import Dataset
import numpy as np
import torch
import torch.nn as nn
import h5py


from .utils import run_once
from ..maptask import MapTaskDataReader
from turn_taking.utils import reset_dir
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)


# TODO: Ensure prosody and full datasets have right dims.
class MapTask(Dataset):
    def __init__(
        self,
        data_dir: str,
        # TODO: Make this configurable when reader bugs are fixed.
        # frame_step_size_ms : int = 10
        num_proc: int = 4,
        num_conversations: int = None,
    ):
        self.base_data_dir = data_dir
        self.feature_dirs = {
            "full": os.path.join(data_dir, "full"),
            "prosody": os.path.join(data_dir, "prosody"),
        }
        self.paths = {"full": None, "prosody": None}

        self.frame_step_size_ms = 10
        self.num_proc = num_proc
        self.num_conversations = num_conversations

        self._download_raw()

    def _download_raw(self):
        logger.debug("DOWNLOADING!")
        # TODO: Need to figure out how to write a lot of data and read
        # it at once.
        reader = MapTaskDataReader(
            num_conversations=self.num_conversations,
            frame_step_size_ms=self.frame_step_size_ms,
            num_proc=self.num_proc,
        )

        reader.prepare_data()
        for variant in ("full", "prosody"):
            if not self._check_exists(variant):
                logger.info(f"Downloading Maptask corpus variant: {variant}")
                logger.debug(f"save dir {self.base_data_dir}")
                reader.setup(
                    variant=variant,
                    save_dir=self.feature_dirs[variant],
                    reset=True,
                )
            else:
                logger.info(f"Loading saved Maptask corpus variant: {variant}")
                reader.load_from_dir(self.feature_dirs[variant])
            self.paths[variant] = reader.data_paths

    def _check_exists(self, variant):
        # TODO: Add the exact size later.
        if not os.path.isdir(self.base_data_dir):
            return False
        feature_dir = self.feature_dirs[variant]
        return (
            os.path.isdir(feature_dir)
            and len(glob.glob(f"{feature_dir}/*.csv")) == self.num_conversations
        )
