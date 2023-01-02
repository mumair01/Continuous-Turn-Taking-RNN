# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-12-21 15:19:06
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-01-02 12:57:39

import sys
import os
import glob

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from .maptask import MapTaskDataReader
from .dsets import MapTaskVADDataset, MapTaskPauseDataset

from utils import get_cache_data_dir


# TODO: Add asserts for function args.
# NOTE: Num workers is 0 here because Maptask reader cannot be pickled.
class MapTaskVADataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir : str,
        sequence_length_ms : int,
        prediction_length_ms : int,
        # frame_step_size_ms,# TODO: Enable after bugfix
        feature_set : str,
        target_participant : str = "f",
        batch_size : int = 32,
        train_split : float = 0.8,
        force_reprocess : bool = False
    ):
        super().__init__()
        self.save_hyperparameters()
        # Vars.
        frame_step_size_ms = 10
        self.sequence_length_ms = sequence_length_ms
        self.prediction_length_ms = prediction_length_ms
        self.target_participant = target_participant
        self.frame_step_size_ms = frame_step_size_ms
        self.data_dir = data_dir
        self.feature_set = feature_set
        self.target_participant = target_participant
        self.train_batch_size = batch_size
        self.val_batch_size = batch_size
        self.train_split = train_split
        self.force_reprocess = force_reprocess

    def prepare_data(self):
        # This will download the dataset if it does not already exist.
        self.dset = MapTaskVADDataset(
            data_dir=self.data_dir,
            sequence_length_ms=self.sequence_length_ms,
            prediction_length_ms=self.prediction_length_ms,
            target_participant=self.target_participant,
            feature_set=self.feature_set,
            force_reprocess=self.force_reprocess
        )

    def setup(self, stage=None):
        # Load the dataset
        dset = self.dset
        # Create the train, val, test splits.
        train_split_size = int(len(dset) * self.train_split)
        self.train_dset, val_dset = random_split(
            dset, [train_split_size, len(dset) - train_split_size])
        # Splitting the initial val set equally into the val and test sets.
        test_split_size = int(len(val_dset) * 0.5)
        self.val_dset, self.test_dset = random_split(
            val_dset, [len(val_dset) - test_split_size, test_split_size]
        )
        # Set up the batch sizes
        if len(self.train_dset) < self.train_batch_size:
            self.train_batch_size = len(self.train_dset)
        if len(self.val_dset) < self.val_batch_size:
            self.val_batch_size = len(self.val_dset)

    def train_dataloader(self):
        return DataLoader(
            self.train_dset,
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset,
            batch_size=self.val_batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            num_workers=0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dset,
            batch_size=1,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            num_workers=0
        )


class MapTaskPauseDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir : str,
        sequence_length_ms : int,
        min_pause_length_ms : int,
        max_future_silence_window_ms : int,
        # frame_step_size_ms, # TODO: Enable after bugfix
        feature_set : str,
        target_participant : str = "f",
        batch_size : int = 32,
        train_split : float = 0.8,
        force_reprocess : bool = False
    ):
        super().__init__()
        self.save_hyperparameters()
        frame_step_size_ms = 10
        self.sequence_length_ms = sequence_length_ms
        self.min_pause_length_ms = min_pause_length_ms
        self.max_future_silence_window_ms = max_future_silence_window_ms
        self.frame_step_size_ms = frame_step_size_ms
        self.data_dir = data_dir
        self.target_participant = target_participant
        self.feature_set = feature_set
        self.train_batch_size = batch_size
        self.val_batch_size = batch_size
        self.train_split = train_split
        self.force_reprocess = force_reprocess

    def prepare_data(self):
        self.dset = MapTaskPauseDataset(
            data_dir=self.data_dir,
            sequence_length_ms=self.sequence_length_ms,
            min_pause_length_ms=self.min_pause_length_ms,
            max_future_silence_window_ms=self.max_future_silence_window_ms,
            s0_participant=self.target_participant,
            feature_set=self.feature_set,
            force_reprocess=self.force_reprocess
        )

    def setup(self, stage=None):
        dset = self.dset

        # Create the train, val, test splits.
        train_split_size = int(len(dset) * self.train_split)
        self.train_dset, val_dset = random_split(
            dset, [train_split_size, len(dset) - train_split_size])
        # Splitting the initial val set equally into the val and test sets.
        test_split_size = int(len(val_dset) * 0.5)
        self.val_dset, self.test_dset = random_split(
            val_dset, [len(val_dset) - test_split_size, test_split_size]
        )
        # Set up the batch sizes
        if len(self.train_dset) < self.train_batch_size:
            self.train_batch_size = len(self.train_dset)
        if len(self.val_dset) < self.val_batch_size:
            self.val_batch_size = len(self.val_dset)


    def train_dataloader(self):
        return DataLoader(
            self.train_dset,
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset,
            batch_size=self.val_batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            num_workers=0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dset,
            batch_size=1,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            num_workers=0
        )




