# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2023-05-31 11:34:34
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-06-04 12:56:36


import sys
import os
import glob

import pandas as pd
from datasets import Dataset
import numpy as np
import torch
import torch.nn as nn
import h5py


from .maptask import MapTaskDataReader
from turn_taking.utils import reset_dir
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)


# TODO: Test this class. Add ability to save pdfs.
class MapTaskPauseDataset:
    HOLD_LABEL = 0
    SHIFT_LABEL = 1
    DATASET_NAME = "pause_dataset"
    PAUSE_DF_COLUMNS = [
        "pauseStartFrameTime",
        "previousSpeaker",
        "pauseEndFrameIndex",
        "nextSpeechFrameIndex",
        "holdShiftLabel",
        "nextSpeaker",
    ]

    def __init__(
        self,
        data_dir,
        sequence_length_ms,
        min_pause_length_ms,
        max_future_silence_window_ms,
        s0_participant,
        feature_set,
        # TODO: Make this configurable when reader bugs are fixed.
        frame_step_size_ms: int = 10,
        force_reprocess=False,
        num_proc: int = 4,
        num_conversations: int = None,
        save_as_csv: bool = False,
    ):
        assert (
            frame_step_size_ms == 10
        ), f"ERROR: Only 10ms frame step size currently supported"

        #### Create the underlying dataset
        # TODO: Add the pos delay as a parameter
        maptask = MapTaskDataReader(
            num_conversations=num_conversations,
            frame_step_size_ms=frame_step_size_ms,
            num_proc=num_proc,
        )
        maptask.prepare_data()
        maptask.setup(
            save_dir=f"{data_dir}/maptask", force_reset=force_reprocess
        )
        self.paths = maptask.data_paths

        # Save vars.
        self.data_dir = data_dir
        self.sequence_length_ms = sequence_length_ms
        self.min_pause_length_ms = min_pause_length_ms
        self.max_future_silence_window_ms = max_future_silence_window_ms
        self.target_participant = s0_participant
        self.frame_step_size_ms = frame_step_size_ms
        self.feature_set = feature_set
        self.force_reprocess = force_reprocess
        self.save_as_csv = save_as_csv
        # Calculated
        self.num_context_frames = int(sequence_length_ms / frame_step_size_ms)
        self.num_pause_frames = int(min_pause_length_ms / frame_step_size_ms)
        self.future_window_frames = int(
            max_future_silence_window_ms / frame_step_size_ms
        )
        # Init.
        self.num_silences = 0
        self.num_holds = 0
        self.num_shifts = 0
        self.num_pauses = 0
        self.length = 0

        #### Setting up directories and h5 file for underlying data storage.
        os.makedirs(data_dir, exist_ok=True)
        self.dset_save_path = os.path.join(data_dir, f"pause_dset.h5")
        self.group_name = (
            f"{self.feature_set}/{self.target_participant}/"
            f"{self.sequence_length_ms}/{self.min_pause_length_ms}/{self.max_future_silence_window_ms}"
        )

        if self.save_as_csv:
            self.csv_dir_path = (
                f"{self.data_dir}/{self.DATASET_NAME}/{self.group_name}/csvs"
            )
            reset_dir(self.csv_dir_path)

        self._prepare()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
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

    def __repr__(self):
        return (
            f"Group naming convention: feature_set/target_participant/"
            "sequence_length_ms/min_pause_length_ms/max_future_silence_window_ms"
            f"min_pause_length_ms: {self.min_pause_length_ms}",
            f"sequence_length_ms: {self.sequence_length_ms}",
            f"max_future_silence_window_ms: {self.max_future_silence_window_ms}",
            f"num_silences: {self.num_silences}",
            f"num_holds: {self.num_holds}",
            f"num_shifts: {self.num_shifts}",
            f"num_pauses: {self.num_pauses}",
            f"s0_participant: {self.s0_participant}",
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
                f"Loading Pause dataset {self.group_name} from disk {self.dset_save_path}"
            )
            f = h5py.File(self.dset_save_path, "r")
            xs_group = f.require_group(f"{self.group_name}/xs")
            ys_group = f.require_group(f"{self.group_name}/ys")
            assert len(xs_group.keys()) == len(ys_group.keys())
            self.length = len(xs_group.keys())
        else:
            logger.debug(
                f"Creating Pause dataset {self.group_name} and saving to {self.dset_save_path}"
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
                xs, ys, pauses_df = self._load_apply_transforms(
                    s0_path, s1_path
                )
                for i in range(xs.shape[0]):
                    # Add this to the appropriate index
                    xs_group.create_dataset(f"{self.length + i}", data=xs[i])
                    ys_group.create_dataset(f"{self.length + i}", data=ys[i])

                # Save the pauses df if needed
                if self.save_as_csv:
                    os.makedirs(self.csv_dir_path, exist_ok=True)
                    dialogue = os.path.splitext(os.path.basename(s0_path))[
                        0
                    ].split(".")[0]
                    pauses_df.to_csv(f"{self.csv_dir_path}/{dialogue}.csv")

                self.length += xs.shape[0]

    #############
    # Methods to apply transformations on the underlying dataset.
    #############

    def _load_apply_transforms(self, s0_path, s1_path):
        # Load the dataframes
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

        # Prepare the pauses df
        pauses_df = self._extract_pauses(
            dfs, self.num_pause_frames, self.future_window_frames
        )

        # Prepare xs and ys based on the pauses df
        s0_s1_df = pd.concat(dfs, axis=1)
        # Remove frametime
        s0_s1_df = s0_s1_df.loc[:, s0_s1_df.columns != "frameTime"]
        assert not s0_s1_df.isnull().values.any()

        xs, ys = list(), list()
        for pause_data in pauses_df.itertuples():
            # Unpack the pause data
            (
                _,
                _,
                previous_speaker,
                idx_after_silence_frames,
                _,
                hold_shift_label,
                next_speaker,
            ) = pause_data
            # Collect features for each speaker equal to sequence length
            # / num context frames.
            if idx_after_silence_frames - self.num_context_frames >= 0:
                start_idx = int(
                    idx_after_silence_frames - self.num_context_frames
                )
                end_idx = int(idx_after_silence_frames)
                x = np.asarray(s0_s1_df.iloc[start_idx:end_idx])
            else:
                num_pad = int(
                    self.num_context_frames - idx_after_silence_frames - 1
                )
                start_idx = 0
                end_idx = int(idx_after_silence_frames + 1)
                x = np.pad(
                    np.asarray(s0_s1_df.iloc[start_idx:end_idx]),
                    [(num_pad, 0), (0, 0)],
                    "constant",
                )
            y = (
                int(previous_speaker),
                int(hold_shift_label),
                int(next_speaker),
            )
            xs.append(x)
            ys.append(y)

        return np.asarray(xs), np.asarray(ys), pauses_df

    # TODO: Check the output.
    def _extract_pauses(
        self, dfs, num_pause_frames: int, future_window_frames: int
    ):
        # Initialize a df to store the pause information
        pauses_df = pd.DataFrame(columns=self.PAUSE_DF_COLUMNS)

        # Obtain frame indices where both speakers are speaking
        sp_va_idxs = [np.where(df["voiceActivity"] == 1)[0] for df in dfs]
        va_idxs = np.union1d(*sp_va_idxs)
        # Obtain the index of the last speaking frame before silences
        va_before_silence_idx = va_idxs[
            np.where(np.diff(va_idxs) > num_pause_frames)
        ]
        # Assign the number of silences
        num_silences = len(va_before_silence_idx)

        # Remove scenarios where both speakers were speaking last i.e., only
        # one speaking could have been speaking before the pause
        va_before_silence_idx = [
            idx
            for idx in va_before_silence_idx
            if not (idx in sp_va_idxs[0] and idx in sp_va_idxs[1])
            and (idx in sp_va_idxs[0] or idx in sp_va_idxs[1])
        ]

        # Next, we want to find all the instances where one (and only one)
        # speaker continues within the next future_window_ms seconds.
        for i, idx in enumerate(va_before_silence_idx):
            # Obtain index of the frame after the specified pause length.
            idx_after_silence = idx + num_pause_frames + 1
            last_idx = idx_after_silence + future_window_frames
            window_vas = [
                np.asarray(df["voiceActivity"])[idx_after_silence:last_idx] == 1
                for df in dfs
            ]
            # Determine the last speaker before the silence
            last_speaker = 0 if dfs[0]["voiceActivity"].iloc[idx] else 1

            # NOTE: Both speakers might start speaking in the future window but we want
            # to make sure that only one of the speakers starts i.e., no overlap.
            # NOTE: 0 = hold, 1 = shift
            next_speaker = np.argmax([window.any() for window in window_vas])
            next_va_idx = (
                np.argmax(window_vas[next_speaker]) + idx_after_silence
            )

            # Determine whether there was a turn hold or shift.
            hold_shift_label = (
                self.HOLD_LABEL
                if last_speaker == next_speaker
                else self.SHIFT_LABEL
            )

            # Add the information to the pause df

            pauses_df.loc[i] = (
                dfs[last_speaker].iloc[idx]["frameTime"],
                last_speaker,
                idx_after_silence,
                next_va_idx,
                hold_shift_label,
                next_speaker,
            )

            # Update the metadata
            self.num_silences += num_silences
            self.num_holds += (
                pauses_df["holdShiftLabel"] == self.HOLD_LABEL
            ).sum()
            self.num_shifts += (
                pauses_df["holdShiftLabel"] == self.SHIFT_LABEL
            ).sum()
            self.num_pauses = self.num_holds + self.num_shifts

        return pauses_df
