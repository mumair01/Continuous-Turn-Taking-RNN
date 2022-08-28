# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-28 10:43:51
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-08-28 11:19:16

import sys
import os
from typing import List, Dict

from data_pipelines.features import OpenSmile

import numpy as np

import torch
import torch.nn as nn

import pytorch_lightning as pl



class ContinuousVoiceActivityDM(pl.LightningDataModule):
    """
    Dataset for voice activity prediction based on Skantze's 2017 paper.

    Performs the following steps:
        1. Extract eGeMaps features for both speakers.
        2. Trim the length of the audio for both speakers to the same length.
        3. Create batches that include the current time step as well as a
            number of context frames, which is determined by the sequence length
            and frame step size.

    Paper Link: https://www.diva-portal.org/smash/get/diva2:1141130/FULLTEXT01.pdf
    """

    def __init__(
        self,
        train_conversation_paths : List[Dict[str,str]],
        val_conversation_paths : List[Dict[str,str]],
        sequence_length_ms : int,
        prediction_length_ms : int,
        frame_step_size_ms : int
    ):
        """
        Assumptions:
            1. Train / val conversation paths are list of maps containing s0
            and s1 as keys and the corresponding audio file path as the value.
        """
        self.train_conversation_paths = train_conversation_paths
        self.val_conversation_paths = val_conversation_paths
        self.sequence_length_ms = sequence_length_ms
        self.prediction_length_ms = prediction_length_ms
        self.frame_step_size_ms = frame_step_size_ms

        self.num_context_frames = int(sequence_length_ms / frame_step_size_ms)
        self.num_target_frames = int(prediction_length_ms / frame_step_size_ms)

        self.smile = OpenSmile(
            feature_set="egemapsv02_50ms",
            feature_level="lld",
            normalize=True,
        )

    ######################## OVERRIDDEN METHODS ##############################

    def prepare_data(self) -> None:
        """
        Prepares the data for the setup method.
        """


    def setup(self, stage: Optional[str] = None) -> None:
        return super().setup(stage)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return super().train_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return super().val_dataloader()

    ######################## ADDITIONAL METHODS ##############################