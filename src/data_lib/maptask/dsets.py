# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-12-22 10:06:06
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-01-01 21:48:03


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

# TODO: Ensure prodody and full datasets have right dims.
class MapTask(Dataset):


    def __init__(
        self,
        data_dir : str,
        # TODO: Make this configurable when reader bugs are fixed.
        # frame_step_size_ms : int = 10
        num_proc : int = 4
    ):
        self.base_data_dir = data_dir
        self.feature_dirs = {
            "full" : os.path.join(data_dir,"full"),
            "prosody" : os.path.join(data_dir,"prosody")
        }
        self.paths = {
            "full" : None,
            "prosody" : None
        }

        self.frame_step_size_ms = 10
        self.num_proc = num_proc

        self._download_raw()

    def _download_raw(self):

        # TODO: Need to figure out how to write a lot of data and read
        # it at once.
        reader = MapTaskDataReader(
            num_conversations=10, # TODO: Change this to None eventually.
            frame_step_size_ms=self.frame_step_size_ms,
            num_proc=self.num_proc

        )
        reader.prepare_data()
        for variant in ("full", "prosody"):
            if not self._check_exists(variant):
                reader.setup(
                    variant=variant,
                    save_dir=self.base_data_dir,
                    reset=False
                )
            else:
                reader.load_from_dir(self.feature_dirs[variant])
            self.paths[variant] = reader.data_paths

    def _check_exists(self, variant):
        # TODO: Add the exact size later.
        if not os.path.isdir(self.base_data_dir):
            return False
        feature_dir = self.feature_dirs[variant]
        return os.path.isdir(feature_dir) and \
            len(glob.glob(f"{feature_dir}/*.csv")) > 0


class MapTaskVADDataset(MapTask):

    DATASET_NAME = "va_dataset"

    def __init__(
        self,
        data_dir : str,
        sequence_length_ms : int,
        prediction_length_ms : int,
        target_participant : int,
        feature_set : str,
        # TODO: Make this configurable when reader bugs are fixed.
        # frame_step_size_ms : int = 10
        force_reprocess : bool = False
    ):
        super().__init__(data_dir)

        # Vars.
        frame_step_size_ms = 10 # TODO: Remove hard coded value layer
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
        os.makedirs(self.save_dir_path,exist_ok=True)
        # H5py management
        self.dset_save_path = os.path.join(data_dir,f"{self.DATASET_NAME}.h5")
        self.group_name = f"{self.feature_set}/{self.target_participant}/:"\
            f"{self.sequence_length_ms}_{self.prediction_length_ms}"

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
        with  h5py.File(self.dset_save_path,'r') as f:
            xs_group = f.require_group(f"{self.group_name}/xs")
            ys_group = f.require_group(f"{self.group_name}/ys")
            return np.asarray(xs_group[f"{idx}"]), np.asarray(ys_group[f"{idx}"])

    ##########
    # Methods to load and save the data
    ##########

    def _prepare(self):
        # If the h5 file already exists, simply load from the file.

        if not self.force_reprocess and os.path.isfile(self.dset_save_path) and \
            self.group_name in h5py.File(self.dset_save_path,'r'):

            f = h5py.File(self.dset_save_path,'r')
            xs_group = f.require_group(f"{self.group_name}/xs")
            ys_group = f.require_group(f"{self.group_name}/ys")
            assert len(xs_group.keys()) == len(ys_group.keys())
            self.length = len(xs_group.keys())
        else:
            # Otherwise, generate all the data and save it in the underlying
            # file at the appropriate group.
            s0_paths = self.paths[self.feature_set][self.target_participant]
            s1_paths = self.paths[self.feature_set]\
                ["f" if self.target_participant == "g" else "g"]

            # Open the file to append.
            f = h5py.File(self.dset_save_path,'a')
            # If the groups already exists, delete it.
            if self.group_name in f:
                del f[self.group_name]

            # Create the groups
            xs_group = f.require_group(f"{self.group_name}/xs")
            ys_group = f.require_group(f"{self.group_name}/ys")

            for (s0_path, s1_path) in zip(s0_paths, s1_paths):
                # Get the xs and ys for these convs.
                xs, ys = self._load_apply_transforms(s0_path, s1_path)
                for i in range(xs.shape[0]):

                    # Add this to the appropriate index
                    xs_group.create_dataset(f"{self.length + i}",data=xs[i])
                    ys_group.create_dataset(f"{self.length + i}",data=ys[i])
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
        dfs = (
            pd.read_csv(s0_path, index_col=0,delimiter=","),
            pd.read_csv(s1_path,index_col=0,delimiter=",")
        )
        min_num_frames = np.min([len(df.index) for df in dfs])

        # Check that there are no null values in the underlying data.
        # Trim the dfs to the same length.
        for df in dfs:
            assert not df.isnull().values.any()
            df = df[:min_num_frames]

        # Check common frametimes
        assert dfs[0]['frameTime'].equals(dfs[1]['frameTime'])

        # Concat to a larger vector
        # TODO: This should be 130 for the full dataset but is currently more.
        s0_s1_df = pd.concat(dfs,axis=1)
        assert not s0_s1_df.isnull().values.any()

        # Generate target VA labels for the target speaker
        va_labels = self._extract_va_target_labels(dfs[0],self.num_target_frames)

        # Prepare sequences for this dialogue.
        num_sequences = int(np.floor(len(dfs[0].index))/self.num_context_frames)
        xs, ys = list(), list()
        for i in range(num_sequences):
            columns = s0_s1_df.columns != 'frameTime'
            start_idx = i * self.num_context_frames
            end_idx = start_idx + self.num_context_frames
            x = np.asarray(
                s0_s1_df.loc[:, columns][ start_idx: end_idx]
            )
            # TODO: Check the y values here.
            # We want the target output to be the last target labels in the
            # sequence.
            y = va_labels[start_idx:end_idx][-1,:]
            xs.append(x)
            ys.append(y)

        return np.asarray(xs), np.asarray(ys)

    def _extract_va_target_labels(self, df : pd.DataFrame, N : int) \
            -> np.ndarray:
        """
        Given a feature df, extract the next N voice activity labels per timestep.
        """
        frame_times = np.asarray(df[["frameTime"]])
        va = np.asarray(df[["voiceActivity"]]).squeeze()
        num_frames = frame_times.shape[0]
        # Check the underlying data
        assert not np.isnan(frame_times).any() and not np.isnan(va).any() and \
            va.shape[0] == num_frames

        # target label shape: Num Frames x N
        labels = np.zeros((num_frames, N))
        for i in range(num_frames):
            # Pad the last labels with 0 if the conversation has ended
            if i + N > num_frames:
                labels[i] = np.concatenate([va[i:], np.zeros(N-(num_frames-i))])
            else:
                labels[i] = va[i:i+N]
        assert not np.isnan(labels).any()
        return labels


# TODO: Test this class. Add ability to save pdfs.
class MapTaskPauseDataset(MapTask):

    HOLD_LABEL = 0
    SHIFT_LABEL = 1
    DATASET_NAME = "pause_dataset"
    PAUSE_DF_COLUMNS = [
        'pauseStartFrameTime',
        'previousSpeaker',
        'pauseEndFrameIndex',
        'nextSpeechFrameIndex',
        'holdShiftLabel',
        'nextSpeaker'
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
        # frame_step_size_ms : int = 10
        force_reprocess = False
    ):

        super().__init__(data_dir)

        frame_step_size_ms = 10 # TODO: Remove hard coded value layer
        self.data_dir = data_dir
        self.sequence_length_ms = sequence_length_ms
        self.min_pause_length_ms = min_pause_length_ms
        self.max_future_silence_window_ms = max_future_silence_window_ms
        self.target_participant = s0_participant
        self.frame_step_size_ms = frame_step_size_ms
        self.feature_set = feature_set
        self.force_reprocess = force_reprocess
        # Calculated
        self.num_context_frames = int(sequence_length_ms / frame_step_size_ms)
        self.num_pause_frames = int(min_pause_length_ms / frame_step_size_ms)
        self.future_window_frames = int(
            max_future_silence_window_ms / frame_step_size_ms)
        # Initialize
        self.num_silences = 0
        self.num_holds = 0
        self.num_shifts = 0
        self.num_pauses = 0
        self.length = 0

        # Create output dir
        self.save_dir_path = data_dir
        os.makedirs(self.save_dir_path,exist_ok=True)
        self.dset_save_path = os.path.join(data_dir,f"{self.DATASET_NAME}.h5")
        self.group_name = f"{self.feature_set}/{self.target_participant}/"\
            f"{self.sequence_length_ms}_{self.min_pause_length_ms}"\
            f"_{self.max_future_silence_window_ms}"

        self._prepare()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx > self.__len__():
            raise Exception
        # NOTE: xs has target speaker features concatenated with non-target speaker features.
        # ys is the previous speaker, hold / shift label, and the next speaker.
        with  h5py.File(self.dset_save_path,'r') as f:
            xs_group = f.require_group(f"{self.group_name}/xs")
            ys_group = f.require_group(f"{self.group_name}/ys")
            return np.asarray(xs_group[f"{idx}"]), np.asarray(ys_group[f"{idx}"])

    def __repr__(self):
        return (
            f"min_pause_length_ms: {self.min_pause_length_ms}" ,
            f"sequence_length_ms: {self.sequence_length_ms}",
            f"max_future_silence_window_ms: {self.max_future_silence_window_ms}",
            f"num_silences: {self.num_silences}",
            f"num_holds: {self.num_holds}",
            f"num_shifts: {self.num_shifts}",
            f"num_pauses: {self.num_pauses}",
            f"s0_participant: {self.s0_participant}"
        )

    def _prepare(self):

        # If the h5 file already exists, simply load from the file.
        if not self.force_reprocess and os.path.isfile(self.dset_save_path) and \
            self.group_name in h5py.File(self.dset_save_path,'r'):
            f = h5py.File(self.dset_save_path,'r')
            xs_group = f.require_group(f"{self.group_name}/xs")
            ys_group = f.require_group(f"{self.group_name}/ys")
            assert len(xs_group.keys()) == len(ys_group.keys())
            self.length = len(xs_group.keys())
        else:
            # Otherwise, generate all the data and save it in the underlying
            # file at the appropriate group.
            s0_paths = self.paths[self.feature_set][self.target_participant]
            s1_paths = self.paths[self.feature_set]\
                ["f" if self.target_participant == "g" else "g"]

            # Open the file to append.
            f = h5py.File(self.dset_save_path,'a')
            # If the groups already exists, delete it.
            if self.group_name in f:
                del f[self.group_name]

            # Create the groups
            xs_group = f.require_group(f"{self.group_name}/xs")
            ys_group = f.require_group(f"{self.group_name}/ys")

            for (s0_path, s1_path) in zip(s0_paths, s1_paths):
                # Get the xs and ys for these convs.
                xs, ys = self._load_apply_transforms(s0_path, s1_path)
                for i in range(xs.shape[0]):

                    # Add this to the appropriate index
                    xs_group.create_dataset(f"{self.length + i}",data=xs[i])
                    ys_group.create_dataset(f"{self.length + i}",data=ys[i])
                self.length += xs.shape[0]

    #############
    # Methods to apply transformations on the underlying dataset.
    #############

    def _load_apply_transforms(self, s0_path, s1_path):
        # Load the dataframes
        dfs = (
            pd.read_csv(s0_path, index_col=0,delimiter=","),
            pd.read_csv(s1_path,index_col=0,delimiter=",")
        )
        min_num_frames = np.min([len(df.index) for df in dfs])

        # Check that there are no null values in the underlying data.
        # Trim the dfs to the same length.
        for df in dfs:
            assert not df.isnull().values.any()
            df = df[:min_num_frames]

        # Check common frametimes
        assert dfs[0]['frameTime'].equals(dfs[1]['frameTime'])

        # Prepare the pauses df
        pauses_df = self._extract_pauses(
            dfs, self.num_pause_frames, self.future_window_frames
        )

        # Prepare xs and ys based on the pauses df
        s0_s1_df = pd.concat(dfs,axis=1)
        # Remove frametime
        s0_s1_df = s0_s1_df.loc[:, s0_s1_df.columns != 'frameTime']
        assert not s0_s1_df.isnull().values.any()

        xs, ys = list(), list()
        for pause_data in pauses_df.itertuples():
            # Unpack the pause data
            _,_, previous_speaker, idx_after_silence_frames, _, \
                hold_shift_label, next_speaker = pause_data
            # Collect features for each speaker equal to sequence length
            # / num context frames.
            if idx_after_silence_frames- self.num_context_frames >= 0:
                start_idx = idx_after_silence_frames - self.num_context_frames
                end_idx = idx_after_silence_frames
                s = np.asarray(s0_s1_df.iloc[start_idx : end_idx])
            else:
                num_pad = self.num_context_frames - idx_after_silence_frames -1
                start_idx = 0
                end_idx = idx_after_silence_frames+1
                x = np.pad(
                    np.asarray(s0_s1_df.iloc[start_idx : end_idx]),
                    [(num_pad,0),(0,0)],'constant'
                )
            y = (int(previous_speaker), int(hold_shift_label), int(next_speaker))
            xs.append(x)
            ys.append(y)

        return np.asarray(xs), np.asarray(ys)

    # TODO: Check the output.
    def _extract_pauses(self,
            dfs,
            num_pause_frames : int,
            future_window_frames : int
        ):
        # Initialize a df to store the pause information
        pauses_df = pd.DataFrame(columns=self.PAUSE_DF_COLUMNS)

        # Obtain frame indices where both speakers are speaking
        va_idxs = np.union1d([np.where(df["voiceActivity"] == 1) for df in dfs])
        # Obtain the index of the last speaking frame before silences
        va_before_silence_idx = va_idxs[np.where(np.diff(va_idxs) > num_pause_frames)]
        # Assign the number of silences
        num_silences = len(va_before_silence_idx)

        # Remove scenarios where both speakers were speaking last i.e., only
        # one speaking could have been speaking before the pause
        va_before_silence_idx = [
            idx for idx in va_before_silence_idx if \
                not (idx in va_idxs[0] and idx in va_idxs[1]) and \
                (idx in va_idxs[0] or idx in va_idxs[1])
        ]

        # Next, we want to find all the instances where one (and only one)
        # speaker continues within the next future_window_ms seconds.
        for i, idx in enumerate(va_before_silence_idx):
            # Obtain index of the frame after the specified pause length.
            idx_after_silence = idx + num_pause_frames + 1
            last_idx = idx_after_silence + future_window_frames
            window_vas = [
                np.asarray(df['voiceActivity'])[idx_after_silence :last_idx] == 1\
                    for df in dfs
            ]
            # Determine the last speaker before the silence
            last_speaker = 0 if dfs[0]['voiceActivity'].iloc[idx] else 1

            # NOTE: Both speakers might start speaking in the future window but we want
            # to make sure that only one of the speakers starts i.e., no overlap.
            # NOTE: 0 = hold, 1 = shift
            next_speaker = np.argmax([window.any() for window in window_vas])
            next_va_idx = np.argmax(window_vas[next_speaker]) + idx_after_silence

            # Determine whether there was a turn hold or shift.
            hold_shift_label = self.HOLD_LABEL if last_speaker == next_speaker \
                else self.SHIFT_LABEL

            # Add the information to the pause df
            pauses_df.iloc[i] = (
                dfs[last_speaker].iloc[idx]['frameTime'],
                last_speaker,
                idx_after_silence,
                next_va_idx,
                hold_shift_label,
                next_speaker
            )

            # Update the metadata
            self.num_silences += num_silences
            self.num_holds += (pauses_df['holdShiftLabel'] == self.HOLD_LABEL).sum()
            self.num_shifts += (pauses_df['holdShiftLabel'] == self.SHIFT_LABEL).sum()
            self.num_pauses = self.num_holds + self.num_shifts

            return pauses_df