# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-28 10:43:51
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-20 17:09:44

import sys
import os
from typing import List, Dict

from data_pipelines.features import OpenSmile, extract_feature_set
from data_pipelines.datasets import load_data
import pandas as pd

from datasets import concatenate_datasets, Dataset
import sklearn

import numpy as np

import torch
import torch.nn as nn

import pytorch_lightning as pl


# Create a vocabulary from all the POS annotation tags
# Documentation to the MapTask POS tags:  https://groups.inf.ed.ac.uk/maptask/interface/expl.html
POS_TAGS = [
    "vb",
    "vbd",
    "vbg",
    "vbn",
    "vbz",
    "nn",
    "nns",
    "np",
    "jj",
    "jjr",
    "jjt",
    "ql",
    "qldt",
    "qlp",
    "rb",
    "rbr",
    "wql",
    "wrb",
    "not",
    "to",
    "be",
    "bem",
    "ber",
    "bez",
    "do",
    "doz",
    "hv",
    "hvz",
    "md",
    "dpr",
    "at",
    "dt",
    "ppg",
    "wdt",
    "ap",
    "cd",
    "od",
    "gen",
    "ex",
    "pd",
    "wps",
    "wpo",
    "pps",
    "ppss",
    "ppo",
    "ppl",
    "ppg2",
    "pr",
    "pn",
    "in",
    "rp",
    "cc",
    "cs",
    "aff",
    "fp",
    "noi",
    "pau",
    "frag",
    "sent"
]


class MapTaskVADDM(pl.LightningDataModule):

    def __init__(
        self,
        sequence_length_ms : int = 60_000,
        prediction_length_ms : int = 3000,
        frame_step_size_ms : int = 10
    ):
        self.sequence_length_ms = sequence_length_ms
        self.prediction_length_ms = prediction_length_ms
        self.frame_step_size_ms = frame_step_size_ms

        self.num_context_frames = sequence_length_ms // frame_step_size_ms
        self.num_target_frames = prediction_length_ms // frame_step_size_ms

        self.xs = []
        self.ys = []

    def prepare_data(self):
        dset_def = load_data(
            dataset="maptask"
        )["full"]
        dset_aud = load_data(
            dataset="maptask",
            variant="audio"
        )["full"]
        dset = self._merge_datasets(dset_def, dset_aud)
        dset = dset.select(range(2)) # For testing
        dset = dset.map(self._extract_audio_features)
        dset = dset.map(self._add_frame_times)
        dset = dset.map(self._extract_voice_activity_annotations)
        dset = dset.map(self._normalize_features)
        dset = dset.map(self._extract_pos_with_delay)

        # print(dset_def)
        # print(len(dset_def))
        # dset = concatenate_datasets([dset_def, dset_aud],axis=1)
        # seen = []
        # for item in dset:
        #     d = item["dialogue"]
        #     p = item["participant"]
        #     x = f"{d}_{p}"
        #     # if x in seen:
        #     #     seen.sort()
        #     #     print(seen)
        #     #     print(x)
        #         # raise Exception()
        #     seen.append(x)
        # print(len(dset))


        # test_dset = test_dset.map(extract_gemaps)
        # # Merge the datasets by dialogue


        # print(maptask_default)
        # print(maptask_audio)

    def setup(self):
        pass

    def _extract_audio_features(self, item):
        # NOTE: This should be 50 ms but this is a bug in the datasets lib.
        item['mono_gemaps'] = extract_feature_set(
            item['audio_paths']['mono'],feature_set="egemapsv02_default")
        return item

    def _add_frame_times(self, item):
        values = np.asarray(item["mono_gemaps"]["values"]).squeeze()
        step = self.frame_step_size_ms / 1000
        frame_times = np.arange(0, step * len(values), step).reshape(-1,1)
        assert frame_times.shape[0] == values.shape[0]
        x = np.column_stack((frame_times,values))
        item["mono_gemaps"]["values"] = x
        item["mono_gemaps"]["features"].insert(0, "frameTime")
        return item

    def _merge_datasets(self, df1, df2):
        assert len(df1) == len(df2)
        df1 = pd.DataFrame(df1)
        df2 = pd.DataFrame(df2)
        merged = pd.merge(df1,df2,how="outer")
        assert len(df1) == len(merged)
        return Dataset.from_pandas(merged)

    def _extract_voice_activity_annotations(self, item):
        frame_times = np.asarray(item["mono_gemaps"]["values"])[:,0]
        va = np.zeros(frame_times.shape[0])
        # Collect the va times

        voiced_times = [(utt["start"], utt["end"]) for utt in item["utterances"]]
        for start_time_s, end_time_s in voiced_times:
            # Obtaining index relative to the frameTimes being considered for
            # which there is voice activity.
            start_idx = np.abs(frame_times - start_time_s).argmin()
            end_idx = np.abs(frame_times-end_time_s).argmin()
            va[start_idx:end_idx+1] = 1
        va = va.reshape(-1,1)
        values = np.asarray(item["mono_gemaps"]["values"])
        x = np.column_stack((values[:,0],va, values[:,1:]))
        item["mono_gemaps"]["values"] = x
        item["mono_gemaps"]["features"].insert(1, "voiceActivity")
        return item

    def _normalize_features(self, item):
        norm_features = [
            "F0semitoneFrom27.5Hz_sma3nz",
            "Loudness_sma3",
            'spectralFlux_sma3'
        ]
        features = np.asarray(item["mono_gemaps"]["features"])
        values = np.asarray(item["mono_gemaps"]["values"])
        assert "F0semitoneFrom27.5Hz_sma3nz" in item["mono_gemaps"]["features"]
        idx = [int(np.where(features == f)[0]) for f in norm_features]
        feats_to_norm = values[:,idx]
        z_norm = lambda v : sklearn.preprocessing.scale(v,axis=1)
        normed = z_norm(feats_to_norm)
        # Add normed features back
        values = np.column_stack((values, normed))
        features = np.append(features,[f"{f}ZNormed"  for f in norm_features])
        item["mono_gemaps"]["features"] = features
        item["mono_gemaps"]["values"] = values
        return item

    def _extract_pos_with_delay(self, item):
        pos_tags_to_idx = {}
        idx_to_pos_tag = {}
        # NOTE: Indices start from 1 here because 0 already represents unknown categories.
        for i,tag in enumerate(POS_TAGS):
            pos_tags_to_idx[tag] = i +1
            idx_to_pos_tag[i+1] = tag

        frame_times = np.asarray(item["mono_gemaps"]["values"])[:,0]
        features = np.asarray(item["mono_gemaps"]["features"])
        values = np.asarray(item["mono_gemaps"]["values"])
        pos = [(utt["end"], utt["pos"]) for utt in item["utterances"] \
            if utt["pos"] in POS_TAGS]


        pos_annotations = np.zeros((frame_times.shape[0]))
        for end_time_s, pos_tag in pos:
            # TODO: the pos delay time should be set somewhere.
            frame_idx = np.abs(frame_times-(end_time_s + 100/1000)).argmin()
            # Convert to integer based on the vocabulary dictionary.
            pos_annotations[frame_idx] = pos_tags_to_idx[pos_tag]
        # The pos annotations should not have any nan values
        assert not np.isnan(pos_annotations).any()
        # This encoder will ignore any unknown tags by replacing them with all zeros.
        onehot_encoder = sklearn.preprocessing.OneHotEncoder(
            sparse=False,handle_unknown="ignore")
        onehot_encoder.fit(np.asarray(list(pos_tags_to_idx.values())).reshape(-1,1))
        encoded_pos = onehot_encoder.transform(pos_annotations.reshape(-1,1))

        values = np.column_stack((values, encoded_pos))
        features = np.append(features,[p for p in POS_TAGS])
        item["mono_gemaps"]["features"] = features
        item["mono_gemaps"]["values"] = values
        return item

class MapTaskPauseDataset(pl.LightningDataModule):
    pass


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


    # def setup(self, stage: Optional[str] = None) -> None:
    #     return super().setup(stage)

    # def train_dataloader(self) -> TRAIN_DATALOADERS:
    #     return super().train_dataloader()

    # def val_dataloader(self) -> EVAL_DATALOADERS:
    #     return super().val_dataloader()

    ######################## ADDITIONAL METHODS ##############################