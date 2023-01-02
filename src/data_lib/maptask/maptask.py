# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-12-20 14:36:46
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-01-02 13:02:59

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
import datasets
from datasets import load_dataset, load_from_disk

import pytorch_lightning as pl

import glob

from utils import reset_dir

import logging
logger = logging.getLogger(__name__)


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


GEMAPS_FREQUENCY_FEATURES = [
    'F0semitoneFrom27.5Hz_sma3nz', # Pitch: logarithmic F0 on a semitone frequency scale, starting at 27.5 Hz (semitone 0)
    # "jitterLocal_sma3nz", # Jitter, deviations in individual consecutive F0 period lengths.
    #  # Formant 1, 2, and 3 frequency, centre frequency of first, second, and third formant
    # "F1frequency_sma3nz",
    # "F2frequency_sma3nz",
    # "F3frequency_sma3nz",
    # "F1bandwidth_sma3nz"
]

GEMAPS_ENERGY_FEATURES = [
    # "shimmerLocaldB_sma3nz", # Shimmer, difference of the peak amplitudes of consecutive F0 periods.
    "Loudness_sma3", # Loudness, estimate of perceived signal intensity from an auditory spectrum.
    # "HNRdBACF_sma3nz" # Harmonics-to-Noise Ratio (HNR), relation of energy in harmonic components to energy in noiselike components.
]

GEMAPS_SPECTRAL_FEATURES = [
    'spectralFlux_sma3',
    # "alphaRatio_sma3", #  Alpha Ratio, ratio of the summed energy from 50–1000 Hz and 1–5 kHz
    # "hammarbergIndex_sma3",  # Hammarberg Index, ratio of the strongest energy peak in the 0–2 kHz region to the strongest peak in the 2–5 kHz region
    # # Spectral Slope 0–500 Hz and 500–1500 Hz, linear regression slope of the logarithmic power spectrum within the two given bands
    # "slope0-500_sma3",
    # "slope500-1500_sma3",
    # # Formant 1, 2, and 3 relative energy, as well as the ratio of the energy of the spectral harmonic
    # # peak at the first, second, third formant’s centre frequency to the energy of the spectral peak at F0.
    # "F1amplitudeLogRelF0_sma3nz",
    # "F2amplitudeLogRelF0_sma3nz",
    # "F3amplitudeLogRelF0_sma3nz",
    # "logRelF0-H1-H2_sma3nz", # Harmonic difference H1–H2, ratio of energy of the first F0 harmonic (H1) to the energy of the second F0 harmonic (H2)
    # "logRelF0-H1-A3_sma3nz" # Harmonic difference H1–A3, ratio of energy of the first F0 harmonic (H1) to the energy of the highest harmonic in the third formant range (A3).
]


RELEVANT_GEMAP_FEATURES = GEMAPS_FREQUENCY_FEATURES + GEMAPS_ENERGY_FEATURES + \
    GEMAPS_SPECTRAL_FEATURES


# Features for both the prosody model.
# TODO: Check whether frameTime is a feature used in the paper.
PROSODY_FEATURES = RELEVANT_GEMAP_FEATURES

# Features for the Full model.
FULL_FEATURES = PROSODY_FEATURES + POS_TAGS

# TODO: Fix single instance crashing issue!
class MapTaskDataReader:

    SUPPORTED_STEP_SIZES = (10,)
    SUPPORTED_VARIANTS = ("full", "prosody")
    # Features to z-normalize.
    NORM_FEATURES = [
        "F0semitoneFrom27.5Hz_sma3nz",
        "Loudness_sma3",
        'spectralFlux_sma3'
    ]
    POS_DELAY_S = 0.1

    def __init__(
        self,
        num_conversations : int = None,
        frame_step_size_ms : int = 10,
        num_proc : int = 4,
    ):
        assert frame_step_size_ms in self.SUPPORTED_STEP_SIZES, \
            f"ERROR: Frame step size must be in {self.SUPPORTED_STEP_SIZES}"

        assert num_proc > 0, \
            f"ERROR: Num process {num_proc} not > 0"

        # Num conversations has to be even since we need to get at least one
        # f and g pair.
        assert num_conversations == None or num_conversations % 2 == 0, \
            f"ERROR: Num conversations specified as {num_conversations}"\
            ", but has to be even"

        self.frame_step_size_ms = frame_step_size_ms
        self.num_conversations = num_conversations
        self.num_proc = num_proc

        if num_conversations != None:
            self.num_proc = min(num_proc, num_conversations)

    @property
    def data_paths(self):
        return self.paths

    def prepare_data(self):
        """
        Download the maptask data if required and prepare for setup.
        """
        dset_def = load_data("maptask")["full"]
        dset_aud = load_data("maptask",variant="audio")["full"]
        # Merge the acoustic and default dataset columns.
        dset = self._merge_datasets(dset_def, dset_aud)

        # Get a subset of the data if required.
        if self.num_conversations != None:
            num_conversations = len(dset) if self.num_conversations > len(dset) \
                else self.num_conversations
            dset = dset.select(range(num_conversations))
        self.dset = dset

    def setup(self, variant, save_dir, reset = False):

        assert variant in self.SUPPORTED_VARIANTS, \
            f"ERROR: Variant {variant} must be in {self.SUPPORTED_VARIANTS}"

        # Create the save path
        os.makedirs(save_dir,exist_ok=True)
        if reset:
            reset_dir(save_dir)

        logger.info(
            f"Processing maptask for variant: {variant}\n"
            f"Number of conversations = {self.num_conversations}\n"
            f"Number of processes = {self.num_proc}"
        )

        dset = self.dset
        num_proc = self.num_proc

        # Extract the audio features, which are the eGeMAPS here
        # Construct the frame times based on the time step ms provided.
        logger.debug(f"Extracting audio features...")
        dset = self._apply_processing_fn(
            "extract_gemaps", self._extract_gemaps, dset,save_dir
        )

        # Extract VA annotations based on the start and end time of utterances.
        logger.debug(f"Extracting voice activity annotations...")
        dset = self._apply_processing_fn(
            "va", self._extract_VA_annotations,dset, save_dir
        )


        logger.debug(f"Performing z-normalization for select features...")
        dset = self._apply_processing_fn(
            "norm", self._normalize_features, dset, save_dir
        )

        if variant == "full":
            # Parts of speech are extracted from the underlying data based on
            # the variant that we need.
            logger.debug("Extracting POS with delay...")
            dset = self._apply_processing_fn(
                "pos", self._extract_pos_with_delay,dset, save_dir
            )

        # With all of the features, we want to separate into f and g and save.
        dset = dset.remove_columns(["utterances", 'audio_paths'])

        # Now, we want to save everything as a csv
        def save_as_csvs(item):
            dialogue, participant = item["dialogue"], item["participant"]
            os.makedirs(f"{save_dir}/{variant}", exist_ok=True)
            path = f"{save_dir}/{variant}/{dialogue}.{participant}.csv"
            try:
                df = pd.DataFrame.from_dict(item["gemaps"])
                df.to_csv(path)
            except:
                logger.debug("*" * 30)
                logger.debug(item["dialogue"], item["participant"])
                for k, v in item["gemaps"].items():
                    logger.debug(k, len(v))

        logger.debug("Saving as csvs...")
        dset.map(save_as_csvs,num_proc=num_proc)

        self.load_from_dir(f"{save_dir}/{variant}")

    def load_from_dir(self, dir_path):
        filepaths = glob.glob("{}/*.csv".format(dir_path))
        paths = defaultdict(lambda : list())
        for path in filepaths:
            for speaker in ("f", "g"):
                if os.path.basename(path).split(".")[1] == speaker:
                    paths[speaker].append(path)
        for speaker in ("f", "g"):
            paths[speaker] = sorted(paths[speaker])
        self.paths = {
            "f" : paths["f"],
            "g" : paths["g"]
        }

    def _apply_processing_fn(self, process, map_fn , dset, save_dir):

        # Load if exists
        process_filepath = f"{save_dir}/temp/{process}"
        if os.path.exists(process_filepath):
            logger.debug(f"Loading intermediate save for: {process}")
            return load_from_disk(process_filepath)
        # Otherwise, apply the map function and save to disk.
        dset = dset.map(map_fn,num_proc=self.num_proc)
        dset.save_to_disk(process_filepath)
        logger.debug(f"Saving intermediate {process}")
        return dset


    def _merge_datasets(self, df1 : Dataset, df2 : Dataset) \
            -> Dataset:
        """
        Given two datasets of the same length, perform an outer merge i.e,
        take the columns of df2 and append then to df1.

        Returns:
            Dataset
        """
        assert len(df1) == len(df2), \
            f"ERROR: Lengths of dfs must be same: {len(df1)} != {len(df2)}"
        df1 = pd.DataFrame(df1)
        df2 = pd.DataFrame(df2)
        merged = pd.merge(df1,df2,how="outer")
        assert len(df1) == len(merged), \
            f"ERROR: Lengths of dfs must be same: {len(df1)} != {len(merged)}"
        return Dataset.from_pandas(merged).remove_columns(['__index_level_0__'])

    def _extract_gemaps(self, item : datasets.arrow_dataset.Example) -> \
            datasets.arrow_dataset.Example:
        """
        Add GeMaps field to item containing mapping from feature to 2D values
        array.
        """
        # NOTE: This should be 50 ms but this is a bug in the datasets lib.

        if self.frame_step_size_ms != 10:
            raise NotImplementedError()

        gemaps = extract_feature_set(
            audio_path = item['audio_paths']['mono'],
            feature_set="egemapsv02_default"
        )
        features = gemaps["features"]
        values = np.asarray(gemaps["values"]).squeeze()
        gemaps_map = dict()
        for i, f in enumerate(features):
            # Keep only the relevant GEMAPS feature
            if f in RELEVANT_GEMAP_FEATURES:
                # Ensure that no value is null
                self._check_null(item, values[:,i])
                gemaps_map[f] = np.asarray(values[:,i])

        # Add the frametimes as a feature
        step_s = self.frame_step_size_ms / 1000
        # NOTE: This is to ensure that the frame_time size is the
        # same as the value arrays.
        frame_times = np.arange(0, step_s * values.shape[0], step_s)[:values.shape[0]]
        assert frame_times.shape[0] == values.shape[0], \
            f"{frame_times.shape} != {values.shape} in dim 0"
        gemaps_map["frameTime"] = np.asarray(frame_times)

        # Put data back into original item.
        item["gemaps"] = gemaps_map
        return item

    def _extract_VA_annotations(self, item : datasets.arrow_dataset.Example) \
            -> datasets.arrow_dataset.Example:
        """
        Extract the VA annotations based on when the start and end times of
        each utterance is for each frame, which is based on the frame step size.
        """

        frame_times = np.asarray(item["gemaps"]["frameTime"])
        va = np.zeros(frame_times.shape[0])
        voiced_times = [(utt["start"], utt["end"]) for utt in item["utterances"]]

        for start_s ,end_s in voiced_times:
            start_idx = np.abs(frame_times - start_s).argmin()
            end_idx = np.abs(frame_times - end_s).argmin()
            # Voice activity is binary - 1 indicates voice.
            va[start_idx:end_idx+1] = 1

        self._check_null(item, va)
        item["gemaps"]["voiceActivity"] = va
        return item

    def _normalize_features(self, item):
        """
        Normalize and add a new k,v pair for features in NORM_FEATURES.
        """
        gemaps_map = item["gemaps"]
        z_norm_func = lambda v : sklearn.preprocessing.scale(v,axis=0)
        for feature in self.NORM_FEATURES:
            values = np.asarray(gemaps_map[feature])
            normed = z_norm_func(values)
            self._check_null(item, normed)
            gemaps_map[f"{feature}ZNormed"] = normed

        item["gemaps"] = gemaps_map
        return item


    def _extract_pos_with_delay(self, item):
        """
        Extract parts of speech and add them with a specific delay.
        """

        # Building a vocabulary for the POS
        pos_tags_to_idx = {}
        idx_to_pos_tag = {}
        # NOTE: Indices start from 1 here because 0 already represents unknown categories.
        for i,tag in enumerate(POS_TAGS):
            pos_tags_to_idx[tag] = i +1
            idx_to_pos_tag[i+1] = tag

        # Adding the pos values with delay
        frame_times = np.asarray(item["gemaps"]["frameTime"])
        pos = [(utt["end"], utt["pos"]) for utt in item["utterances"] \
            if utt["pos"] in POS_TAGS]
        pos_annotations = np.zeros((frame_times.shape[0]))
        for end_time_s, pos_tag in pos:
            # TODO: the pos delay time should be set somewhere.
            frame_idx = np.abs(frame_times-(end_time_s + self.POS_DELAY_S)).argmin()
            # Convert to integer based on the vocabulary dictionary.
            pos_annotations[frame_idx] = pos_tags_to_idx[pos_tag]
        # The pos annotations should not have any nan values
        self._check_null(item, pos_annotations)

        # One-hot encoding the POS values
        # This encoder will ignore any unknown tags by replacing them with all zeros.
        onehot_encoder = sklearn.preprocessing.OneHotEncoder(
            sparse=False,handle_unknown="ignore")
        onehot_encoder.fit(np.asarray(list(pos_tags_to_idx.values())).reshape(-1,1))
        encoded_pos = onehot_encoder.transform(pos_annotations.reshape(-1,1))

        # Add the POS as features
        pos_map = {}
        for i, pos in enumerate(onehot_encoder.categories_[0]):
            pos_map[idx_to_pos_tag[pos]] = encoded_pos[:,i]
        item["gemaps"].update(pos_map)

        return item

    def _check_null(self, item,  arr):
        if np.isnan(arr).any():
            d, p = item["dialogue"], item["participant"]
            msg = f"Null values encountered in gemaps: {d}, {p}"
            logger.debug(msg)
            raise Exception(msg)

