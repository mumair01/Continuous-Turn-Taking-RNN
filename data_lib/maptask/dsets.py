# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-12-22 10:06:06
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-22 10:36:22


import sys
import os

import pandas as pd
from datasets import Dataset
import numpy as np
import torch
import torch.nn as nn


class MapTaskVADDataset(Dataset):
    def __init__(
        self,
        paths,
        sequence_length_ms,
        prediction_length_ms,
        target_participant,
        frame_step_size_ms,
    ):
        self.paths = paths
        # Vars.
        self.sequence_length_ms = sequence_length_ms
        self.prediction_length_ms = prediction_length_ms
        self.target_participant = target_participant
        self.frame_step_size_ms = frame_step_size_ms
        # Calculated
        self.num_context_frames = int(sequence_length_ms / frame_step_size_ms)
        self.num_target_frames = int(prediction_length_ms / frame_step_size_ms)

        self.xs = []
        self.ys = []
        for s0_path, s1_path in paths:
            s0_feature_df = pd.read_csv(s0_path, index_col=0,delimiter=",")
            s1_feature_df = pd.read_csv(s1_path,index_col=0,delimiter=",")
            self._load_data(s0_feature_df, s1_feature_df)
        assert len(self.xs) == len(self.ys)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        if idx > self.__len__():
            raise Exception
        return self.xs[idx], self.ys[idx]

    def _load_data(self, s0_feature_df, s1_feature_df):
        # Extract the voice activity labels for s0 as the target labels
        s0_target_labels_df = self._extract_voice_activity_labels(
            s0_feature_df,self.num_target_frames)
        # Make sure none of the dfs have any nan values
        assert not s0_feature_df.isnull().values.any() and \
            not s1_feature_df.isnull().values.any() and \
            not s0_target_labels_df.isnull().values.any()
        # Trim the dataframes to the same length
        min_num_frames = np.min([len(s0_feature_df.index),len(s1_feature_df.index)])
        s0_feature_df = s0_feature_df[:min_num_frames]
        s1_feature_df = s1_feature_df[:min_num_frames]
        s0_target_labels_df = s0_target_labels_df[:min_num_frames]
        # Make sure they all have common frametimes
        assert s0_feature_df['frameTime'].equals(s1_feature_df['frameTime'])
        assert s0_feature_df['frameTime'].equals(s0_target_labels_df['frameTime'])
        s0_s1_df = pd.concat([s0_feature_df,s1_feature_df],axis=1)
        assert not s0_s1_df.isnull().values.any()
        # Determine the number of sequences for this dialogue
        num_sequences = int(np.floor(len(s0_feature_df.index))/self.num_context_frames)
        for i in range(num_sequences):
            x = np.asarray(s0_s1_df.loc[:,s0_s1_df.columns != 'frameTime'][i * \
                self.num_context_frames : (i * self.num_context_frames) \
                    + self.num_context_frames])
            y = np.asarray(s0_target_labels_df.loc[:,s0_target_labels_df.columns\
                    != 'frameTime'][i * self.num_context_frames : \
                        (i * self.num_context_frames) + self.num_context_frames])[-1,:]
            self.xs.append(x)
            self.ys.append(y)

    def _extract_voice_activity_labels(self, feature_df, N):
        # TODO: FIx the delimiter
        feature_df = feature_df[["frameTime","voiceActivity"]]
        assert not feature_df.isnull().values.any()
        frame_times_ms = np.asarray(feature_df["frameTime"])
        voice_activity_annotations = np.asarray(feature_df["voiceActivity"])
        assert frame_times_ms.shape[0] == voice_activity_annotations.shape[0]
        labels = np.zeros((frame_times_ms.shape[0],N)) # target label shape: Num Frames x N
        for i in range(len(frame_times_ms)):
            # Pad the last labels with 0 if the conversation has ended
            if i + N > len(frame_times_ms):
                concat = np.concatenate(
                    [voice_activity_annotations[i:],
                    np.zeros(N - (len(frame_times_ms)-i))])
                labels[i] = concat
            else:
                labels[i] = voice_activity_annotations[i:i+N]
        labels_df = pd.DataFrame(labels)
        labels_df.insert(0,"frameTime",frame_times_ms)
        assert not labels_df.isnull().values.any()
        return labels_df


class MapTaskPauseDataset(Dataset):

    def __init__(
        self,
        paths,
        sequence_length_ms,
        min_pause_length_ms,
        max_future_silence_window_ms,
        s0_participant,
        frame_step_size_ms
    ):
        self.paths = paths
        self.sequence_length_ms = sequence_length_ms
        self.min_pause_length_ms = min_pause_length_ms
        self.max_future_silence_window_ms = max_future_silence_window_ms
        self.s0_participant = s0_participant
        self.frame_step_size_ms = frame_step_size_ms
        # Calculated
        self.num_context_frames = int(sequence_length_ms / frame_step_size_ms)
        self.num_pause_frames = int(min_pause_length_ms / frame_step_size_ms)
        self.future_window_frames = int(
            max_future_silence_window_ms / frame_step_size_ms)
        # Data Storage vars.
        self.xs = []
        self.ys = []
        self.num_silences = 0
        self.num_holds = 0
        self.num_shifts = 0
        self.num_pauses = 0
        for s0_path, s1_path in paths:
            s0_feature_df = pd.read_csv(s0_path, index_col=0,delimiter=",")
            s1_feature_df = pd.read_csv(s1_path,index_col=0,delimiter=",")
            assert len(s0_feature_df) == len(s1_feature_df)
            pauses_df = self._load_data(s0_feature_df, s1_feature_df)
            self.__prepare_items(s0_feature_df, s1_feature_df,pauses_df)

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

    def __prepare_items(self,s0_feature_df, s1_feature_df, pauses_df):
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

    def __prepare_pauses_df(self, s0_feature_df, s1_feature_df):
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


