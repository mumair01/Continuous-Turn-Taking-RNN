# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-12-20 14:36:46
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-22 10:48:39

import sys
import os
from typing import List, Dict

from data_pipelines.features import OpenSmile, extract_feature_set
from data_pipelines.datasets import load_data
import pandas as pd

from datasets import concatenate_datasets, Dataset
import sklearn
from functools import partial

import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn

import pytorch_lightning as pl

import glob

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


COMMON_FEATURES = [
    "frameTime", "voiceActivity", "Loudness_sma3",
    "F0semitoneFrom27.5Hz_sma3nz", "F0semitoneFrom27.5Hz_sma3nzZNormed",
    "spectralFlux_sma3ZNormed", "Loudness_sma3ZNormed"
]

FULL_FEATURES = COMMON_FEATURES + POS_TAGS

PROSODY_FEATURES = COMMON_FEATURES

class MapTaskDataReader:


    def __init__(
        self,
        frame_step_size_ms : int = 10,
        num_conversations : int = None
    ):
        self.frame_step_size_ms = frame_step_size_ms
        self.num_conversations = num_conversations

    @property
    def data_paths(self):
        f_paths = sorted(self.paths["f"])
        g_paths = sorted(self.paths["g"])
        return {
            "f" : f_paths,
            "g" : g_paths
        }

    def load_from_dir(self, dir_path):
        filepaths = glob.glob("{}/*.csv".format(dir_path))
        paths = defaultdict(lambda : list())
        for path in filepaths:
            if os.path.basename(path).split(".")[1] == "f":
                paths["f"].append(path)
            elif os.path.basename(path).split(".")[1] == "g":
                paths["g"].append(path)
        self.paths = paths
        return paths

    def prepare_data(self):
        dset_def = load_data(
            dataset="maptask"
        )["full"]
        dset_aud = load_data(
            dataset="maptask",
            variant="audio"
        )["full"]
        dset = self._merge_datasets(dset_def, dset_aud)
        if self.num_conversations != None:
            dset = dset.select(range(self.num_conversations))
        self.dset = dset

    def setup(self, variant, save_dir):

        print(
            f"Processing maptask for variant: {variant}\n"
            f"Number of conversations = {self.num_conversations}"
        )

        dset = self.dset
        num_proc = 4
        print(f"Extracting audio features...")
        dset = dset.map(self._extract_audio_features, num_proc=num_proc)
        print(f"Adding frame times...")
        dset = dset.map(self._add_frame_times,num_proc=num_proc)
        print(f"Extracting voice activity annotations...")
        dset = dset.map(self._extract_voice_activity_annotations,num_proc=num_proc)
        print(f"Normalizing features")
        dset = dset.map(self._normalize_features,num_proc=num_proc)
        if variant == "full":
            print("Extracting POS with delay...")
            dset = dset.map(self._extract_pos_with_delay,num_proc=num_proc)
        dset = dset.rename_column("mono_gemaps", "processed").remove_columns([
            "utterances", 'audio_paths', '__index_level_0__'
        ])
        f = dset.filter(lambda item: item["participant"] == "f").sort("dialogue")
        g = dset.filter(lambda item: item["participant"] == "g").sort("dialogue")

        path = f"{save_dir}/{variant}"
        os.makedirs(path, exist_ok=True)

        def convert_item_to_df_and_save(item):
            values = np.asarray(item['processed']["values"])
            features = item["processed"]["features"]
            df = pd.DataFrame(values,columns=features)

            df = df[FULL_FEATURES if variant == "full" else PROSODY_FEATURES]
            dialogue, participant = item["dialogue"], item["participant"]
            filepath = f"{path}/{dialogue}.{participant}.csv"
            df.to_csv(f"{filepath}")

        self.paths = {}
        print(f"Saving items to directory: {path}")
        f.map(convert_item_to_df_and_save,num_proc=num_proc)
        g.map(convert_item_to_df_and_save,num_proc=num_proc)
        return self.load_from_dir(path)

    def _extract_audio_features(self, item):
        # NOTE: This should be 50 ms but this is a bug in the datasets lib.
        item['mono_gemaps'] = extract_feature_set(
            item['audio_paths']['mono'],feature_set="egemapsv02_default")
        return item

    def _add_frame_times(self, item):
        values = np.asarray(item["mono_gemaps"]["values"]).squeeze()
        step = self.frame_step_size_ms / 1000
        frame_times = np.arange(0, step * len(values), step).reshape(-1,1)
        assert frame_times.shape[0] == values.shape[0], \
            f"ERROR: {frame_times.shape} != {values.shape} in the first dim"
        x = np.column_stack((frame_times,values))
        item["mono_gemaps"]["values"] = x
        item["mono_gemaps"]["features"].insert(0, "frameTime")
        return item

    def _merge_datasets(self, df1, df2):
        assert len(df1) == len(df2), \
            f"ERROR: Lengths of dfs must be same: {len(df1)} != {len(df2)}"
        df1 = pd.DataFrame(df1)
        df2 = pd.DataFrame(df2)
        merged = pd.merge(df1,df2,how="outer")
        assert len(df1) == len(merged), \
            f"ERROR: Lengths of dfs must be same: {len(df1)} != {len(merged)}"
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
        assert "F0semitoneFrom27.5Hz_sma3nz" in item["mono_gemaps"]["features"], \
            f"ERROR: Feature not found"
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
        assert not np.isnan(pos_annotations).any(), \
            f"ERROR: POS annotations must not have null values"
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

if __name__ == "__main__":

    reader = MapTaskDataReader(
        frame_step_size_ms=10,
        num_conversations=50
    )
    reader.prepare_data()
    reader.setup(
        variant="full",
        save_dir="./output"
    )