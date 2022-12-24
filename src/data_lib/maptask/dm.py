# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-12-21 15:19:06
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-24 15:39:23

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
# Either remove it from the dataset or keep workers at 0.
class MapTaskVADataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir,
        sequence_length_ms,
        prediction_length_ms,
        frame_step_size_ms,
        feature_set,
        target_participant = "f",
        batch_size = 32,
        train_split = 0.8
    ):
        super().__init__()
        # Vars.
        self.sequence_length_ms = sequence_length_ms
        self.prediction_length_ms = prediction_length_ms
        self.target_participant = target_participant
        self.frame_step_size_ms = frame_step_size_ms
        self.data_dir = data_dir
        self.feature_set = feature_set
        self.target_participant = target_participant
        self.batch_size = batch_size
        self.train_split = train_split

    def prepare_data(self):
        # This will download the dataset if it does not already exist.
        MapTaskVADDataset(
            data_dir=self.data_dir,
            sequence_length_ms=self.sequence_length_ms,
            prediction_length_ms=self.prediction_length_ms,
            target_participant=self.target_participant,
            feature_set=self.feature_set,
            force_reprocesses=False
        )

    def setup(self, stage=None):
        # Load the dataset
        dset = MapTaskVADDataset(
            data_dir=self.data_dir,
            sequence_length_ms=self.sequence_length_ms,
            prediction_length_ms=self.prediction_length_ms,
            target_participant=self.target_participant,
            feature_set=self.feature_set,
            force_reprocesses=False
        )


        # Create the train, val, test splits.
        train_split_size = int(len(dset) * self.train_split)
        self.train_dset, val_dset = random_split(
            dset, [train_split_size, len(dset) - train_split_size])
        # Splitting the initial val set equally into the val and test sets.
        test_split_size = int(len(val_dset) * 0.5)
        self.val_dset, self.test_dset = random_split(
            val_dset, [len(val_dset) - test_split_size, test_split_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset,
            batch_size=self.batch_size,
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
        data_dir,
        sequence_length_ms,
        min_pause_length_ms,
        max_future_silence_window_ms,
        # frame_step_size_ms, # TODO: Enable after bugfix
        target_participant,
        feature_set,
        batch_size = 32,
        train_split = 0.8
    ):
        super().__init__()
        frame_step_size_ms = 10
        self.sequence_length_ms = sequence_length_ms
        self.min_pause_length_ms = min_pause_length_ms
        self.max_future_silence_window_ms = max_future_silence_window_ms
        self.frame_step_size_ms = frame_step_size_ms
        self.data_dir = data_dir
        self.target_participant = target_participant
        self.feature_set = feature_set
        self.batch_size = batch_size
        self.train_split = train_split

    def prepare_data(self):
        MapTaskPauseDataset(
            data_dir=self.data_dir,
            sequence_length_ms=self.sequence_length_ms,
            min_pause_length_ms=self.min_pause_length_ms,
            max_future_silence_window_ms=self.max_future_silence_window_ms,
            s0_participant=self.target_participant,
            feature_set=self.feature_set,
            force_reprocess=False
        )

    def setup(self, stage=None):
        dset = MapTaskPauseDataset(
            data_dir=self.data_dir,
            sequence_length_ms=self.sequence_length_ms,
            min_pause_length_ms=self.min_pause_length_ms,
            max_future_silence_window_ms=self.max_future_silence_window_ms,
            s0_participant=self.target_participant,
            feature_set=self.feature_set,
            force_reprocess=False
        )

        # Create the train, val, test splits.
        train_split_size = int(len(dset) * self.train_split)
        self.train_dset, val_dset = random_split(
            dset, [train_split_size, len(dset) - train_split_size])
        # Splitting the initial val set equally into the val and test sets.
        test_split_size = int(len(val_dset) * 0.5)
        self.val_dset, self.test_dset = random_split(
            val_dset, [len(val_dset) - test_split_size, test_split_size]
        )


    def train_dataloader(self):
        return DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset,
            batch_size=self.batch_size,
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




