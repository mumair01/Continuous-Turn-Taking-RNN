# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-12-21 12:49:48
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-21 13:22:12


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


import os
import glob
import sys
import shutil
import argparse
from functools import reduce, partial
from multiprocessing import Pool
import time
import tqdm
from vardefs import *
from features import *
from utils import *


class MapTaskReader:

    def __init__(self):
        pass

    def prepare_data(self, variant=None, **kwargs):
        self._prepare_skantze_2017_data(**kwargs)

    def _prepare_skantze_2017_data(self, dialogue_names_split, output_dir, num_threads=4, feature_set="full"):
        start_time = time.time()
        assert os.path.isdir(output_dir)

        # os.makedirs(CORRECTED_GEMAPS_DIR,exist_ok=True)
        # for dialogue_name in dialogue_names_split:
        #     for participant in PARTICIPANT_LABELS_MAPTASK:
        #         verify_correct_gemaps(GEMAPS_PATH, dialogue_name, participant,CORRECTED_GEMAPS_DIR)

        print("Running pipeline for {} dialogues, each for participants {}...".format(
            len(dialogue_names_split),PARTICIPANT_LABELS_MAPTASK))
        collected_args = []
        for dialogue_name in dialogue_names_split:
            for participant in PARTICIPANT_LABELS_MAPTASK:
                collected_args.append((dialogue_name,participant, output_dir,feature_set))
        print("Using {} threads...".format(num_threads))
        with Pool(num_threads) as pool:
            results = list(tqdm.tqdm(pool.imap(
                self._extract_skantze_2017_features_star, collected_args),
                total=len(collected_args),desc="Extracting Features"))
        elapsed_time = time.time() - start_time
        print("Elapsed time: {:.3f} seconds".format(elapsed_time))
        print("Results saved to {}".format(output_dir))
        print("Completed!")

    def _extract_skantze_2017_features_star(self, args):
        self._extract_skantze_2017_features(*args)

    def _extract_skantze_2017_features(self, dialogue_name, participant, output_dir,feature_set):
        voice_activity_df = get_voice_activity_annotations(
            dialogue_name, participant)
        pitch_df = extract_pitch(dialogue_name, participant)
        power_df = extract_power(dialogue_name, participant)
        spectral_flux_df = extract_spectral_flux(dialogue_name, participant)
        if feature_set == "prosody":
            # Does not include pos features
            result_df = reduce(lambda x,y: pd.merge(
            x,y, on='frameTime', how='outer'),
            [voice_activity_df,pitch_df,power_df,spectral_flux_df])
        else:
            pos_df = extract_pos_annotations_with_delay(dialogue_name, participant)
            result_df = reduce(lambda x,y: pd.merge(
                x,y, on='frameTime', how='outer'),
                [voice_activity_df,pitch_df,power_df,spectral_flux_df,pos_df])
        result_df.to_csv("{}/{}.{}.skantze_2017_features.{}.csv".format(
            output_dir,dialogue_name,participant,feature_set))

if __name__ == "__main__":
    # Obtain args
    FEATURE_SETS = ["full","prosody"]
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--out_dir", dest="out_dir", type=str, required=True,
    #                     help="Path to the output directory")
    # parser.add_argument("--set", dest="feat_set", type=str, required=True,
    #                     help="One of {}".format(FEATURE_SETS))
    # args = parser.parse_args()
    # assert args.feat_set in FEATURE_SETS
    # Create the output directory if it does not exist
    os.makedirs("./test",exist_ok=True)
    # TODO: For now - read all the timed-unit files and extract features for all
    dialogue_names = [os.path.basename(p).split(".")[0] for p in glob.glob("{}/*.xml".format(TIMED_UNIT_PATHS))]

    reader = MapTaskReader()
    reader.prepare_data(
        dialogue_names_split=dialogue_names,
        output_dir="./test",
        num_threads=5,
        feature_set="/Users/muhammadumair/Documents/Repositories/mumair01-repos/TRP-Detection/data/maptask"
    )