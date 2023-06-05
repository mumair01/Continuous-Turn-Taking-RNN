# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-12-20 14:36:46
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-06-05 09:35:28

import sys
import os
import glob
from collections import defaultdict
import json

import pandas as pd
import numpy as np
import torch.nn as nn
import sklearn
from datasets import Dataset, load_from_disk


from data_pipelines.features import OpenSmile, extract_feature_set
from data_pipelines.datasets import load_data

from turn_taking.dsets.utils import reset_dir
from turn_taking.dsets.maptask.datasets.constants import MapTaskConstants

from typing import List, Callable, Dict

import logging

logger = logging.getLogger(__name__)

# TODO: Update the documentation and make the path names variables.


# Step sizes that this data reader can currently support.
_SUPPORTED_STEP_SIZES = (10,)
# The datasets that can currently be created.
_SUPPORTED_VARIANTS = ("full", "prosody")
# Features to z-normalize.
_NORM_FEATURES = [
    "F0semitoneFrom27.5Hz_sma3nz",
    "Loudness_sma3",
    "spectralFlux_sma3",
]
# # The delay added to parts of speech to simulate incoming data.
# _POS_DELAY_S = 0.1


# TODO: Fix single instance crashing issue!
class MapTaskDataReader:
    """
    Provides an abstraction over the maptask dataset, creates all relevant
    features, and provides a mechanism for saving and loading relevant data.
    """

    def __init__(
        self,
        num_conversations: int = None,
        frame_step_size_ms: int = 10,
        pos_delay_s: float = 0.1,
        num_proc: int = 4,
    ):
        """
        Parameters
        ----------
        num_conversations : int, optional
            Number of total maptask conversations to use, if None, uses all
            available conversations. Note that a single conversation includes
            both the f and g component of the conversation (which is actually 2
            distinct audio files), by default None
        frame_step_size_ms : int, optional
            Step size for each frame. Only 10 is currently supported, by default 10
        num_proc : int, optional
            Number of processes to use when generating dataset, by default 4
        """
        assert (
            frame_step_size_ms in _SUPPORTED_STEP_SIZES
        ), f"ERROR: Frame step size must be in {_SUPPORTED_STEP_SIZES}"

        assert num_proc > 0, f"ERROR: Num process {num_proc} not > 0"

        self.frame_step_size_ms = frame_step_size_ms
        self.pos_delay_s = pos_delay_s
        self.num_conversations = num_conversations
        self.num_proc = num_proc
        self.save_dir = None
        self.paths = defaultdict(lambda: dict)

        if num_conversations != None:
            self.num_proc = min(num_proc, num_conversations)

    @property
    def data_paths(self) -> List[str]:
        """
        Obtain the paths of the underlying data files
        NOTE: setup must have been called
        """
        return dict(self.paths)

    def prepare_data(self) -> None:
        """
        Download the maptask data if required and prepare for setup.

        Attributes set
        --------------
        self.paths, which can be obtained via data_paths
        """
        dset_def = load_data("maptask")["full"]
        dset_aud = load_data("maptask", variant="audio")["full"]
        # Merge the acoustic and default dataset columns.
        dset = self._merge_datasets(dset_def, dset_aud)

        # Obtain a subset of the data if required
        if self.num_conversations != None:
            # Get unique dialogue names and select corresponding f and g files
            dialogues = sorted(list(set(dset["dialogue"])))
            dialogues = (
                dialogues
                if self.num_conversations > len(dialogues)
                else dialogues[: self.num_conversations]
            )
            dset = dset.filter(lambda item: item["dialogue"] in dialogues)
        else:
            # Update num conversations
            self.num_conversations = len(sorted(list(set(dset["dialogue"]))))

        self.dset = dset

    def setup(self, save_dir: str, force_reset=False):
        """
        Create all relevant features that are required from the underlying
        dataset for both the full and prosody variants.

        Parameters
        ----------
        save_dir : str
            Path to the output directory where the prepared datasets should be
            saved.
        reset : bool, optional
            If reset, overwrites the existing output directory if it exists,
            by default False
        """

        # Create paths for this variant

        cache_dir = f"{save_dir}/cache"
        os.makedirs(cache_dir, exist_ok=True)
        for variant in _SUPPORTED_VARIANTS:
            variant_dir = f"{save_dir}/{variant}"
            os.makedirs(variant_dir, exist_ok=True)

            logger.info(
                f"Processing maptask for variant: {variant}; "
                f"Number of conversations = {self.num_conversations}; "
                f"Number of processes = {self.num_proc}"
            )
            logger.info(force_reset)
            # Simply load the appropriate number of paths from disk.
            if not self._should_reset(variant_dir) and not force_reset:
                self.paths[variant] = self._load_from_dir(
                    variant_dir, self.num_conversations
                )
                logger.info(f"Loading cached data from: {variant_dir}")
                continue
            # Otherwise, we need to reset the saved dir. and cache dir and reconstruct.
            else:
                reset_dir(variant_dir)
                if self._should_reset_cache(cache_dir):
                    reset_dir(cache_dir)

            dset = self.dset
            num_proc = self.num_proc

            # Extract the audio features, which are the eGeMAPS here
            # Construct the frame times based on the time step ms provided.
            logger.debug(f"Extracting audio features...")
            dset = self._apply_processing_fn(
                "extract_gemaps", self._extract_gemaps, dset, cache_dir
            )

            # Extract VA annotations based on the start and end time of utterances.
            logger.debug(f"Extracting voice activity annotations...")
            dset = self._apply_processing_fn(
                "va", self._extract_VA_annotations, dset, cache_dir
            )

            logger.debug(f"Performing z-normalization for select features...")
            dset = self._apply_processing_fn(
                "norm", self._normalize_features, dset, cache_dir
            )

            if variant == "full":
                # Parts of speech are extracted from the underlying data based on
                # the variant that we need.
                logger.debug("Extracting POS with delay...")
                dset = self._apply_processing_fn(
                    "pos", self._extract_pos_with_delay, dset, cache_dir
                )

            # Add metadata to the cache dir
            with open(f"{cache_dir}/metadata.json", "w") as outfile:
                json.dump(
                    {"num_conversations": self.num_conversations}, outfile
                )

            # With all of the features, we want to separate into f and g and save.
            dset = dset.remove_columns(["utterances", "audio_paths"])

            # Now, we want to save everything as a csv
            def save_as_csvs(item):
                dialogue, participant = item["dialogue"], item["participant"]
                path = f"{variant_dir}/{dialogue}.{participant}.csv"
                try:
                    df = pd.DataFrame.from_dict(item["gemaps"])
                    df.to_csv(path)
                except:
                    logger.debug("*" * 30)
                    logger.debug(item["dialogue"], item["participant"])
                    for k, v in item["gemaps"].items():
                        logger.debug(k, len(v))

            logger.debug("Saving as csvs...")
            dset.map(save_as_csvs, num_proc=num_proc)
            self.paths[variant] = self._load_from_dir(
                variant_dir, self.num_conversations
            )

    def _should_reset(self, variant_path: str) -> bool:
        if os.path.isdir(variant_path):
            paths = self._load_from_dir(variant_path)
            return (
                self.num_conversations > len(paths["f"])
                if self.num_conversations
                else False  # Do not reset if num conversations is none.
            )
        return True

    def _should_reset_cache(self, cache_dir: str) -> bool:
        meta_path = f"{cache_dir}/metadata.json"
        if not os.path.isfile(meta_path):
            return True
        with open(meta_path) as f:
            data = json.load(f)
            return data["num_conversations"] != self.num_conversations

    def _load_from_dir(
        self, dir_path: str, num_conversations: int = None
    ) -> Dict:
        """
        Loads a dataset that was previously prepared by MapTaskDataReader
        from disk.

        Parameters
        ----------
        dir_path : str
            Path to the directory where the dataset was saved.

        Attributes Set
        -------------
        self.paths: List[str]
            Sets the paths to the data files, loaded from disk.
        """
        filepaths = glob.glob("{}/*.csv".format(dir_path))
        paths = defaultdict(lambda: list())
        for path in filepaths:
            for speaker in ("f", "g"):
                if os.path.basename(path).split(".")[1] == speaker:
                    paths[speaker].append(path)
        for speaker in ("f", "g"):
            paths[speaker] = sorted(paths[speaker])
        if num_conversations == None:
            num_conversations = len(paths["f"])
        return {
            "f": paths["f"][:num_conversations],
            "g": paths["g"][:num_conversations],
        }

    def _apply_processing_fn(
        self,
        process: str,
        map_fn: Callable,
        dset: Dataset,
        cache_dir: str,
    ) -> Dataset:
        """
        Apply a map function to the given directory and save the results in
        the specified output directory.

        Parameters
        ----------
        process : str
            Name of the process i.e, unique identifier used to create a temp. dir.
        map_fn : Callable
            Function to be applied on the entire dataset.
        dset : Dataset
            Dataset obj.
        save_dir : str
            Path to the output directory.
        Returns
        -------
        Dataset
            Datasets after the map_fn has been applied
        """
        # Load if exists
        process_filepath = f"{cache_dir}/{process}"
        if os.path.exists(process_filepath):
            logger.debug(f"Loading intermediate save for: {process}")
            return load_from_disk(process_filepath)
        # Otherwise, apply the map function and save to disk.
        dset = dset.map(map_fn, num_proc=self.num_proc)
        dset.save_to_disk(process_filepath)
        logger.debug(f"Saving intermediate {process}")
        return dset

    def _merge_datasets(self, df1: Dataset, df2: Dataset) -> Dataset:
        """
        Given two datasets of the same length, perform an outer merge i.e,
        take the columns of df2 and append then to df1.

        Parameters
        ----------
        df1 : Dataset
        df2 : Dataset

        Returns
        -------
        Dataset
        """
        assert len(df1) == len(
            df2
        ), f"ERROR: Lengths of dfs must be same: {len(df1)} != {len(df2)}"
        df1 = pd.DataFrame(df1)
        df2 = pd.DataFrame(df2)
        merged = pd.merge(df1, df2, how="outer")
        assert len(df1) == len(
            merged
        ), f"ERROR: Lengths of dfs must be same: {len(df1)} != {len(merged)}"
        # return Dataset.from_pandas(merged).remove_columns(["__index_level_0__"])
        return Dataset.from_pandas(merged)

    def _extract_gemaps(self, item: Dict) -> Dict:
        """
        Add GeMaps field to item containing mapping from feature to 2D values
        array.
        NOTE: This should be 50 ms but this is a bug in the datasets lib.

        Parameters
        ----------
        item : Dict
            A single item from the maptask dataset

        Returns
        -------
        Dict
            Modified item from the maptask dataset

        Raises
        ------
        NotImplementedError
           Raised if the frame step size is unavailable
        """

        if self.frame_step_size_ms != 10:
            raise NotImplementedError()

        gemaps = extract_feature_set(
            audio_path=item["audio_paths"]["mono"],
            feature_set="egemapsv02_default",
        )
        features = gemaps["features"]
        values = np.asarray(gemaps["values"]).squeeze()
        gemaps_map = dict()
        for i, f in enumerate(features):
            # Keep only the relevant GEMAPS feature
            if f in MapTaskConstants.PROSODY_FEATURES:
                # Ensure that no value is null
                self._check_null(item, values[:, i])
                gemaps_map[f] = np.asarray(values[:, i])

        # Add the frametimes as a feature
        step_s = self.frame_step_size_ms / 1000
        # NOTE: This is to ensure that the frame_time size is the
        # same as the value arrays.
        frame_times = np.arange(0, step_s * values.shape[0], step_s)[
            : values.shape[0]
        ]
        assert (
            frame_times.shape[0] == values.shape[0]
        ), f"{frame_times.shape} != {values.shape} in dim 0"
        gemaps_map["frameTime"] = np.asarray(frame_times)

        # Put data back into original item.
        item["gemaps"] = gemaps_map
        return item

    def _extract_VA_annotations(self, item: Dict) -> Dict:
        """
        Extract the VA annotations based on when the start and end times of
        each utterance is for each frame, which is based on the frame step size.

        Parameters
        ----------
        item : Dict
            A single item from the maptask dataset

        Returns
        -------
        Dict
            Modified item from the maptask dataset
        """

        frame_times = np.asarray(item["gemaps"]["frameTime"])
        va = np.zeros(frame_times.shape[0])
        voiced_times = [
            (utt["start"], utt["end"]) for utt in item["utterances"]
        ]

        for start_s, end_s in voiced_times:
            start_idx = np.abs(frame_times - start_s).argmin()
            end_idx = np.abs(frame_times - end_s).argmin()
            # Voice activity is binary - 1 indicates voice.
            va[start_idx : end_idx + 1] = 1

        self._check_null(item, va)
        item["gemaps"]["voiceActivity"] = va
        return item

    def _normalize_features(self, item: Dict) -> Dict:
        """
        Normalize and add a new k,v pair for features in NORM_FEATURES.

        Parameters
        ----------
        item : Dict
            A single item from the maptask dataset

        Returns
        -------
        Dict
            Modified item from the maptask dataset
        """
        gemaps_map = item["gemaps"]
        z_norm_func = lambda v: sklearn.preprocessing.scale(v, axis=0)
        for feature in _NORM_FEATURES:
            values = np.asarray(gemaps_map[feature])
            normed = z_norm_func(values)
            self._check_null(item, normed)
            gemaps_map[f"{feature}ZNormed"] = normed

        item["gemaps"] = gemaps_map
        return item

    def _extract_pos_with_delay(self, item):
        """
        Extract parts of speech and add them with a specific delay.

        Parameters
        ----------
        item : Dict
            A single item from the maptask dataset

        Returns
        -------
        Dict
            Modified item from the maptask dataset
        """

        # Building a vocabulary for the POS
        pos_tags_to_idx = {}
        idx_to_pos_tag = {}
        # NOTE: Indices start from 1 here because 0 already represents unknown categories.
        for i, tag in enumerate(MapTaskConstants.POS_TAGS):
            pos_tags_to_idx[tag] = i + 1
            idx_to_pos_tag[i + 1] = tag

        # Adding the pos values with delay
        frame_times = np.asarray(item["gemaps"]["frameTime"])
        pos = [
            (utt["end"], utt["pos"])
            for utt in item["utterances"]
            if utt["pos"] in MapTaskConstants.POS_TAGS
        ]
        pos_annotations = np.zeros((frame_times.shape[0]))
        for end_time_s, pos_tag in pos:
            # TODO: the pos delay time should be set somewhere.
            frame_idx = np.abs(
                frame_times - (end_time_s + self.pos_delay_s)
            ).argmin()
            # Convert to integer based on the vocabulary dictionary.
            pos_annotations[frame_idx] = pos_tags_to_idx[pos_tag]
        # The pos annotations should not have any nan values
        self._check_null(item, pos_annotations)

        # One-hot encoding the POS values
        # This encoder will ignore any unknown tags by replacing them with all zeros.
        onehot_encoder = sklearn.preprocessing.OneHotEncoder(
            sparse=False, handle_unknown="ignore"
        )
        onehot_encoder.fit(
            np.asarray(list(pos_tags_to_idx.values())).reshape(-1, 1)
        )
        encoded_pos = onehot_encoder.transform(pos_annotations.reshape(-1, 1))

        # Add the POS as features
        pos_map = {}
        for i, pos in enumerate(onehot_encoder.categories_[0]):
            pos_map[idx_to_pos_tag[pos]] = encoded_pos[:, i]
        item["gemaps"].update(pos_map)

        return item

    def _check_null(self, item: Dict, arr: np.asarray):
        """
        Check whether there is a null value in the given array for the given item

        Parameters
        ----------
        item : Dict
        arr : np.asarray

        Raises
        ------
        Exception
            Raised if there is any null value in arr.
        """
        if np.isnan(arr).any():
            d, p = item["dialogue"], item["participant"]
            msg = f"Null values encountered in gemaps: {d}, {p}"
            logger.debug(msg)
            raise Exception(msg)
