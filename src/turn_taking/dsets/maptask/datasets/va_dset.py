# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2023-05-31 11:34:38
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-05-31 16:53:03


import sys
import os
import glob

import pandas as pd
from datasets import Dataset
import numpy as np
import torch
import torch.nn as nn
import h5py


from .base import MapTask
from turn_taking.utils import reset_dir
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)


class MapTaskVADDataset(MapTask):
    DATASET_NAME = "va_dataset"

    def __init__(
        self,
        data_dir: str,
        sequence_length_ms: int,
        prediction_length_ms: int,
        target_participant: int,
        feature_set: str,
        # TODO: Make this configurable when reader bugs are fixed.
        # frame_step_size_ms : int = 10
        force_reprocess: bool = False,
        num_proc: int = 4,
        num_conversations: int = None,
    ):
        super().__init__(data_dir, num_proc, num_conversations)
        # Vars.
        frame_step_size_ms = 10  # TODO: Remove hard coded value layer
        self.data_dir = data_dir
        self.sequence_length_ms = sequence_length_ms
        self.prediction_length_ms = prediction_length_ms
        self.target_participant = target_participant
        self.frame_step_size_ms = frame_step_size_ms
        self.feature_set = feature_set
        self.force_reprocess = force_reprocess
        # Calculated
        self.num_context_frames = int(sequence_length_ms / frame_step_size_ms)
        self.num_target_frames = int(prediction_length_ms / frame_step_size_ms)

        self.length = 0
        # Create output dir
        self.save_dir_path = data_dir
        os.makedirs(self.save_dir_path, exist_ok=True)
        # H5py management
        self.dset_save_path = os.path.join(
            self.save_dir_path, f"{self.DATASET_NAME}.{self.feature_set}.h5"
        )
        self.group_name = (
            f"{self.feature_set}/{self.target_participant}/"
            f"{self.sequence_length_ms}/{self.prediction_length_ms}"
        )

        self._prepare()

    def __len__(self):
        """Obtain the length of the total dataset"""
        return self.length

    def __getitem__(self, idx):
        """
        Get the item in the specified index.
        Since this dataset is storing a lot of data, it is actually reading
        each index from an underlying h5 file.
        """
        if idx > self.__len__():
            raise Exception
        # NOTE: xs has target speaker features concatenated with non-target speaker features.
        # ys is the previous speaker, hold / shift label, and the next speaker.
        with h5py.File(self.dset_save_path, "r") as f:
            xs_group = f.require_group(f"{self.group_name}/xs")
            ys_group = f.require_group(f"{self.group_name}/ys")
            return np.asarray(xs_group[f"{idx}"]), np.asarray(
                ys_group[f"{idx}"]
            )

    ##########
    # Methods to load and save the data
    ##########

    def _prepare(self):
        # If the h5 file already exists, simply load from the file.

        if (
            not self.force_reprocess
            and os.path.isfile(self.dset_save_path)
            and self.group_name in h5py.File(self.dset_save_path, "r")
        ):
            logger.debug(
                f"Loading VA dataset {self.group_name} from disk {self.dset_save_path}"
            )
            f = h5py.File(self.dset_save_path, "r")
            xs_group = f.require_group(f"{self.group_name}/xs")
            ys_group = f.require_group(f"{self.group_name}/ys")
            assert len(xs_group.keys()) == len(ys_group.keys())
            self.length = len(xs_group.keys())
        else:
            logger.debug(
                f"Creating VA dataset {self.group_name} and saving to {self.dset_save_path}"
            )
            # Otherwise, generate all the data and save it in the underlying
            # file at the appropriate group.
            s0_paths = self.paths[self.feature_set][self.target_participant]
            s1_paths = self.paths[self.feature_set][
                "f" if self.target_participant == "g" else "g"
            ]

            # Open the file to append.
            f = h5py.File(self.dset_save_path, "a")
            # If the groups already exists, delete it.
            if self.group_name in f:
                del f[self.group_name]

            # Create the groups
            xs_group = f.require_group(f"{self.group_name}/xs")
            ys_group = f.require_group(f"{self.group_name}/ys")

            logger.info(f"Using {len(s0_paths)} file pair to generate dataset")

            for s0_path, s1_path in tqdm(zip(s0_paths, s1_paths)):
                # Get the xs and ys for these convs.
                xs, ys = self._load_apply_transforms(s0_path, s1_path)
                for i in range(xs.shape[0]):
                    # Add this to the appropriate index
                    xs_group.create_dataset(f"{self.length + i}", data=xs[i])
                    ys_group.create_dataset(f"{self.length + i}", data=ys[i])

                self.length += xs.shape[0]
        f.close()

    #############
    # Methods to apply transformations on the underlying dataset.
    #############

    def _load_apply_transforms(self, s0_path, s1_path):
        """
        For the given s0 and s1 paths, returns the xs and the ys.
        """
        # Load the dataframes
        logger.debug(
            f"Applying transformations to file pair: {s0_path, s1_path}"
        )
        dfs = (
            pd.read_csv(s0_path, index_col=0, delimiter=","),
            pd.read_csv(s1_path, index_col=0, delimiter=","),
        )
        min_num_frames = np.min([len(df.index) for df in dfs])

        # Check that there are no null values in the underlying data.
        # Trim the dfs to the same length.
        for df in dfs:
            assert not df.isnull().values.any()
            df = df[:min_num_frames]

        # Check common frametimes
        assert dfs[0]["frameTime"].equals(dfs[1]["frameTime"])

        # Concat to a larger vector
        # TODO: This should be 130 for the full dataset but is currently more.
        s0_s1_df = pd.concat(dfs, axis=1)

        assert not s0_s1_df.isnull().values.any()

        # Generate target VA labels for the target speaker
        va_labels = self._extract_va_target_labels(
            dfs[0], self.num_target_frames
        )

        # Prepare sequences for this dialogue.
        num_sequences = int(
            np.floor(len(dfs[0].index)) / self.num_context_frames
        )
        xs, ys = list(), list()
        logger.debug(f"Preparing {num_sequences} VA sequences")
        for i in range(num_sequences):
            columns = ~s0_s1_df.columns.isin(["frameTime", "voiceActivity"])
            start_idx = i * self.num_context_frames
            end_idx = start_idx + self.num_context_frames
            x = np.asarray(s0_s1_df.loc[:, columns][start_idx:end_idx])
            # TODO: Check the y values here.
            # We want the target output to be the last target labels in the
            # sequence.
            y = va_labels[start_idx:end_idx][-1, :]
            xs.append(x)
            ys.append(y)

        return np.asarray(xs), np.asarray(ys)

    def _extract_va_target_labels(self, df: pd.DataFrame, N: int) -> np.ndarray:
        """
        Given a feature df, extract the next N voice activity labels per timestep.
        """
        frame_times = np.asarray(df[["frameTime"]])
        va = np.asarray(df[["voiceActivity"]]).squeeze()
        num_frames = frame_times.shape[0]
        # Check the underlying data
        assert (
            not np.isnan(frame_times).any()
            and not np.isnan(va).any()
            and va.shape[0] == num_frames
        )

        # target label shape: Num Frames x N
        labels = np.zeros((num_frames, N))
        for i in range(num_frames):
            # Pad the last labels with 0 if the conversation has ended
            if i + N > num_frames:
                labels[i] = np.concatenate(
                    [va[i:], np.zeros(N - (num_frames - i))]
                )
            else:
                labels[i] = va[i : i + N]
        assert not np.isnan(labels).any()
        return labels
