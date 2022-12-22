# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-12-21 15:19:06
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-22 13:40:19

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

# TODO: Add asserts for function args.
class MapTaskVADataModule(pl.LightningDataModule):

    def __init__(
        self,
        sequence_length_ms,
        prediction_length_ms,
        frame_step_size_ms,
        data_dir = None,
        variant = None,
        save_dir = None,
        target_participant = "f",
        batch_size = 32
    ):
        super().__init__()
        # Vars.
        self.sequence_length_ms = sequence_length_ms
        self.prediction_length_ms = prediction_length_ms
        self.target_participant = target_participant
        self.frame_step_size_ms = frame_step_size_ms
        self.data_dir = data_dir
        self.variant = variant
        self.save_dir = save_dir
        self.target_participant = target_participant
        self.batch_size = batch_size

    def prepare_data(self):
        reader = MapTaskDataReader(
            frame_step_size_ms=self.frame_step_size_ms,
            num_conversations=None
        )
        if self.data_dir == None:
            reader.prepare_data()
            self.paths = reader.setup(
                variant=self.variant,
                save_dir=self.save_dir
            )
        elif os.path.isdir(self.data_dir):
            self.paths = reader.load_from_dir(self.data_dir)
        else:
            raise Exception()


    def setup(self, stage=None):
        # NOTE: Must load all data
        if self.target_participant == "f":
            paths = list(zip(self.paths["f"], self.paths["g"]))
        else:
            paths = list(zip(self.paths["g"], self.paths["f"]))

        dset = MapTaskVADDataset(
            paths=paths,
            sequence_length_ms=self.sequence_length_ms,
            prediction_length_ms=self.prediction_length_ms,
            target_participant=self.target_participant,
            frame_step_size_ms=self.frame_step_size_ms
        )

        train_set_size = int(len(dset) * 0.8)
        valid_set_size = len(dset) - train_set_size
        train_set, valid_set = random_split(
            dset, [train_set_size, valid_set_size]
        )
        test_set_size = int(train_set_size * 0.2)
        train_set_size = len(train_set) - test_set_size
        train_set, test_set = random_split(
            train_set, [train_set_size, test_set_size]
        )

        self.train = train_set
        self.val = valid_set
        self.test = test_set

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=1,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=0
        )


class MapTaskPauseDataModule(pl.LightningDataModule):

    def __init__(
        self,
        sequence_length_ms,
        min_pause_length_ms,
        max_future_silence_window_ms,
        frame_step_size_ms,
        data_dir = None,
        variant = None,
        save_dir = None,
        target_participant="f"
    ):
        super().__init__()
        self.sequence_length_ms = sequence_length_ms
        self.min_pause_length_ms = min_pause_length_ms
        self.max_future_silence_window_ms = max_future_silence_window_ms
        self.frame_step_size_ms = frame_step_size_ms
        self.data_dir = data_dir
        self.variant = variant
        self.save_dir = save_dir
        self.target_participant = target_participant

    def prepare_data(self):
        reader = MapTaskDataReader(
            frame_step_size_ms=self.frame_step_size_ms,
            num_conversations=None
        )
        if self.data_dir == None:
            reader.prepare_data()
            self.paths = reader.setup(
                variant=self.variant,
                save_dir=self.save_dir
            )
        elif os.path.isdir(self.data_dir):
            self.paths = reader.load_from_dir(self.data_dir)

        self.target_participant = self.target_participant

    def setup(self, stage=None):
        # NOTE: Must load all data
        if self.target_participant == "f":
            paths = list(zip(self.paths["f"], self.paths["g"]))
        else:
            paths = list(zip(self.paths["g"], self.paths["f"]))

        dset = MapTaskPauseDataset(
            paths=paths,
            sequence_length_ms=self.sequence_length_ms,
            min_pause_length_ms=self.min_pause_length_ms,
            max_future_silence_window_ms=self.max_future_silence_window_ms,
            s0_participant=self.target_participant,
            frame_step_size_ms=self.frame_step_size_ms
        )

        train_set_size = int(len(dset) * 0.8)
        valid_set_size = len(dset) - train_set_size
        train_set, valid_set = random_split(
            dset, [train_set_size, valid_set_size]
        )
        test_set_size = int(train_set_size * 0.2)
        train_set_size = len(train_set) - test_set_size
        train_set, test_set = random_split(
            train_set, [train_set_size, test_set_size]
        )

        self.train = train_set
        self.val = valid_set
        self.test = test_set

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=32,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=32,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=1,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=0
        )




