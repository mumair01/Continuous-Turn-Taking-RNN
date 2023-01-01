# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-12-22 10:06:06
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-01-01 10:55:30


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
            num_conversations=None, # TODO: Change this to None eventually.
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
        force_reprocesses : bool = False
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
        self.force_reprocess = force_reprocesses
        # Calculated
        self.num_context_frames = int(sequence_length_ms / frame_step_size_ms)
        self.num_target_frames = int(prediction_length_ms / frame_step_size_ms)

        # Create output dir
        self.save_dir_path = data_dir
        os.makedirs(self.save_dir_path,exist_ok=True)
        self.dset_save_path = os.path.join(data_dir,f"{self.DATASET_NAME}.h5")
        self._prepare()

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        if idx > self.__len__():
            raise Exception
        # NOTE: xs has target speaker features concatenated with non-target speaker features.
        # ys is the previous speaker, hold / shift label, and the next speaker.
        return self.xs[idx], self.ys[idx]

    #####
    # Methods to load and save the data
    #####

    def _prepare(self):
        if self._verify_dataset() and not self.force_reprocess:
            self.xs, self.ys = self._load_from_disk()
        else:
            paths = self.paths[self.feature_set]
            s0_paths = paths[self.target_participant]
            s1_paths = paths["f" if self.target_participant == "g" else "g"]
            for s0_path, s1_path in zip(s0_paths, s1_paths):
                self._load_apply_transforms(s0_path, s1_path)

            # Save the dataset
            self._save()
        # Length of xs and ys must be the same.
        assert len(self.xs) == len(self.ys)

    def _save(self, xs, ys):
        """
        Save xs and ys in the relevant group on a conversation level.
        """
        with h5py.File(self.dset_save_path,'a') as hf:
            key = f"{self.sequence_length_ms}_{self.prediction_length_ms}"
            # Delete if exists
            if self.force_reprocess and self._verify_dataset():
                del hf[self.feature_set][self.target_participant][key]

            group = f"{self.feature_set}/{self.target_participant}/{key}"
            hf.create_dataset(f"{group}/xs", data=np.asarray(self.xs))
            hf.create_dataset(f"{group}/ys", data=np.asarray(self.ys))

    def _load_from_disk(self):
        """
        Load the data from the dset path.
        """
        with h5py.File(self.dset_save_path,'r') as hf:
            key = f"{self.sequence_length_ms}_{self.prediction_length_ms}"
            group = hf[self.feature_set][self.target_participant][key]
            return group["xs"][:], group["ys"][:]

    def _verify_dataset(self):
        """
        Checks the h5 dataset to ensure all of the data exists.
        """

        if not (os.path.isdir(self.save_dir_path) and \
                os.path.exists(self.dset_save_path)):
            return False

        with h5py.File(self.dset_save_path,'r') as hf:
            key = f"{self.sequence_length_ms}_{self.prediction_length_ms}"
            return self.feature_set in hf and \
                self.target_participant in hf[self.feature_set] and \
                key in hf[self.feature_set][self.target_participant]

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

        # Concat to a larget vector
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
        return xs, ys

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


class MapTaskPauseDataset(MapTask):

    HOLD_LABEL = 0
    SHIFT_LABEL = 1

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

        super().__init__(f"{data_dir}/maptask")

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

        self.num_silences = 0
        self.num_holds = 0
        self.num_shifts = 0
        self.num_pauses = 0

        self.save_dir_path = data_dir
        self.h5_filename = "pause_dataset.h5"
        os.makedirs(self.save_dir_path,exist_ok=True)

        self._prepare()

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        if idx > self.__len__():
            raise Exception
        # NOTE: xs has target speaker features concatenated with non-target speaker features.
        # ys is the previous speaker, hold / shift label, and the next speaker.
        return self.xs[idx], self.ys[idx]

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
        if self._check_pause_exists() and not self.force_reprocess:
            self._load_from_disk()
        else:
            paths = self.full_paths if self.feature_set == "full" \
                else self.prosody_paths

            s0_paths = paths[self.target_participant]
            s1_paths = paths["f" if self.target_participant == "g" else "g"]
            self.xs = []
            self.ys = []
            for s0_path, s1_path in zip(s0_paths, s1_paths):
                self._load_apply_transforms(s0_path, s1_path)
            # Save the dataset
            self._save()
        assert len(self.xs) == len(self.ys)

    def _load_apply_transforms(self, s0_path, s1_path):
        # TODO: Add multiprocessing here.
        s0_feature_df = pd.read_csv(s0_path, index_col=0,delimiter=",")
        s1_feature_df = pd.read_csv(s1_path,index_col=0,delimiter=",")
        # Trim the dataframes to the same length
        min_num_frames = np.min([len(s0_feature_df.index),len(s1_feature_df.index)])
        s0_feature_df = s0_feature_df[:min_num_frames]
        s1_feature_df = s1_feature_df[:min_num_frames]
        assert len(s0_feature_df) == len(s1_feature_df)
        pauses_df = self._prepare_pauses_df(s0_feature_df, s1_feature_df)
        self._prepare_items(s0_feature_df, s1_feature_df,pauses_df)

    def _check_pause_exists(self):
        if os.path.isdir(self.save_dir_path) and \
                os.path.exists(os.path.join(self.save_dir_path,self.h5_filename)):
            with h5py.File(f"{self.save_dir_path}/{self.h5_filename}", 'r') as hf:
                return self.feature_set in hf and \
                        self.target_participant in hf[self.feature_set] and \
                        f"{self.sequence_length_ms}_{self.min_pause_length_ms}"\
                            f"_{self.max_future_silence_window_ms}" in \
                            hf[self.feature_set][self.target_participant]
        return False

    def _save(self):
        group = f"{self.feature_set}/{self.target_participant}/"\
            f"{self.sequence_length_ms}_{self.min_pause_length_ms}"\
            f"_{self.max_future_silence_window_ms}"
        # Save the xs and ys
        with h5py.File(f"{self.save_dir_path}/{self.h5_filename}", 'a') as hf:
            if self.force_reprocess:
                del hf[self.feature_set][self.target_participant]\
                    [f"{self.sequence_length_ms}_{self.min_pause_length_ms}"\
                    f"_{self.max_future_silence_window_ms}"]
            hf.create_dataset(f"{group}/xs", data=np.asarray(self.xs))
            hf.create_dataset(f"{group}/ys", data=np.asarray(self.ys))

    def _load_from_disk(self):
        with h5py.File(f"{self.save_dir_path}/{self.h5_filename}", 'r') as hf:
            group = hf[self.feature_set][self.target_participant]
            group = group[f"{self.sequence_length_ms}_{self.min_pause_length_ms}"\
                f"_{self.max_future_silence_window_ms}"]
            self.xs = group["xs"][:]
            self.ys = group["ys"][:]

    def _prepare_items(self,s0_feature_df, s1_feature_df, pauses_df):
        # Collect the data for both models
        s0_s1_df = pd.concat([s0_feature_df.loc[:,s0_feature_df.columns \
            != 'frameTime'],s1_feature_df.loc[:,s1_feature_df.columns != 'frameTime']],axis=1)
        s1_s0_df = pd.concat([s1_feature_df.loc[:,s1_feature_df.columns \
            != 'frameTime'],s0_feature_df.loc[:,s0_feature_df.columns != 'frameTime']],axis=1)
        for pause_data in pauses_df.itertuples():
            _,_, previous_speaker, idx_after_silence_frames, _, \
                hold_shift_label, next_speaker = pause_data
            idx_after_silence_frames = int(idx_after_silence_frames)
            # Collect features for each speaker equal to sequence length / num context frames.
            if idx_after_silence_frames- self.num_context_frames >= 0:
                x_s0 = np.asarray(s0_s1_df.iloc[idx_after_silence_frames-\
                    self.num_context_frames:idx_after_silence_frames])
                x_s1 = np.asarray(s1_s0_df.iloc[idx_after_silence_frames-\
                    self.num_context_frames:idx_after_silence_frames])
            else:
                num_pad = self.num_context_frames - idx_after_silence_frames -1
                x_s0 = np.pad(np.asarray(s0_s1_df.iloc[0:idx_after_silence_frames+1])\
                    ,[(num_pad,0),(0,0)],'constant')
                x_s1 = np.pad(np.asarray(s1_s0_df.iloc[0:idx_after_silence_frames+1])\
                    ,[(num_pad,0),(0,0)],'constant')
            self.xs.append((x_s0,x_s1))
            self.ys.append((int(previous_speaker), int(hold_shift_label), int(next_speaker)))

    def _prepare_pauses_df(self, s0_feature_df, s1_feature_df):
        # Obtain frame indices where both speakers are speaking.
        s0_va_idxs =  np.where(s0_feature_df['voiceActivity'] == 1)[0]
        s1_va_idxs =  np.where(s1_feature_df['voiceActivity'] == 1)[0]
        va_idxs = np.union1d(s0_va_idxs,s1_va_idxs)
        # Obtain index of last speaking frame before silences
        speak_before_silence_frames_idx = \
            va_idxs[np.where(np.diff(va_idxs) > self.num_pause_frames)]
        self.num_silences += len(speak_before_silence_frames_idx)
        # Remove scenarios where both speakers were speaking last i.e., only
        # one speaking could have been speaking before te pause
        speak_before_silence_frames_idx = [idx for idx in speak_before_silence_frames_idx \
            if not (idx in s0_va_idxs and idx in s1_va_idxs) \
                and (idx in s0_va_idxs or idx in s1_va_idxs)]
        # Next, we want to find all the instances where one (and only one)
        # speaker continues within the next future_window_ms seconds.
        pauses_df = pd.DataFrame(columns=['pauseStartFrameTime', 'previousSpeaker',
            'pauseEndFrameIndex', 'nextSpeechFrameIndex', 'holdShiftLabel', 'nextSpeaker'])
        for i,idx in enumerate(speak_before_silence_frames_idx):
            # Obtain index of the frame after the specified pause length.
            idx_after_silence_frames = idx + self.num_pause_frames + 1
            # Get the voice activity in the specified future window
            s0_window_va = np.asarray((s0_feature_df['voiceActivity'])[
                idx_after_silence_frames:idx_after_silence_frames+self.future_window_frames] == 1)
            s1_window_va = np.asarray((s1_feature_df['voiceActivity'])[
                idx_after_silence_frames:idx_after_silence_frames+self.future_window_frames] == 1)
            # Determine the last speaker before silence
            last_participant = 0 if s0_feature_df['voiceActivity'].iloc[idx] else 1
            # NOTE: Both speakers might start speaking in the future window but we want
            # to make sure that only one of the speakers starts i.e., no overlap.
            # NOTE: 0 = hold, 1 = shift
            # Condition 1: Speaker 0 is next.
            if s0_window_va.any() and not s1_window_va.any():
                next_va_idx = np.argmax(s0_window_va) + idx_after_silence_frames
                hold_shift_label = self.HOLD_LABEL if last_participant == 0 \
                    else self.SHIFT_LABEL
                pauses_df.loc[i] = (
                    s0_feature_df.iloc[idx]['frameTime'], last_participant,
                     idx_after_silence_frames, next_va_idx, hold_shift_label,0)
            # Condition 2: Speaker 1 is next.
            elif s1_window_va.any() and not s0_window_va.any():
                next_va_idx = np.argmax(s1_window_va) + idx_after_silence_frames
                hold_shift_label = self.HOLD_LABEL if last_participant == 1 \
                    else self.SHIFT_LABEL
                pauses_df.loc[i] = (
                    s1_feature_df.iloc[idx]['frameTime'], last_participant,
                    idx_after_silence_frames, next_va_idx, hold_shift_label,1)
        # Update the shift / hold values
        self.num_holds += (pauses_df['holdShiftLabel'] == self.HOLD_LABEL).sum()
        self.num_shifts += (pauses_df['holdShiftLabel'] == self.SHIFT_LABEL).sum()
        self.num_pauses = self.num_holds + self.num_shifts
        # Save the pauses df if save dir provided
        # if self.save_dir != None:
        #     pauses_df.to_csv("{}/{}_pauses_df.csv".format(self.save_dir, dialogue))
        return pauses_df