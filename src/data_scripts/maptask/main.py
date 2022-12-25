# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-05-18 09:58:15
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-06-15 17:57:12


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

# NOTE: Can be made more efficient by putting each extraction into its own thread.
def extract_skantze_2017_features(dialogue_name, participant, output_dir,feature_set):
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

def extract_skantze_2017_features_star(args):
    extract_skantze_2017_features(*args)

def run_pipeline(dialogue_names_split, output_dir, num_threads=4, feature_set="full"):
    start_time = time.time()
    assert os.path.isdir(output_dir)
    print("Running pipeline for {} dialogues, each for participants {}...".format(
        len(dialogue_names_split),PARTICIPANT_LABELS_MAPTASK))
    collected_args = []
    for dialogue_name in dialogue_names_split:
        for participant in PARTICIPANT_LABELS_MAPTASK:
            collected_args.append((dialogue_name,participant, output_dir,feature_set))
    print("Using {} threads...".format(num_threads))
    with Pool(num_threads) as pool:
        results = list(tqdm.tqdm(pool.imap(
            extract_skantze_2017_features_star, collected_args),
            total=len(collected_args),desc="Extracting Features"))
    elapsed_time = time.time() - start_time
    print("Elapsed time: {:.3f} seconds".format(elapsed_time))
    print("Results saved to {}".format(output_dir))
    print("Completed!")


# TODO: Add an example command here.
if __name__ == "__main__":
    # Obtain args
    FEATURE_SETS = ["full","prosody"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", dest="out_dir", type=str, required=True,
                        help="Path to the output directory")
    parser.add_argument("--set", dest="feat_set", type=str, required=True,
                        help="One of {}".format(FEATURE_SETS))
    args = parser.parse_args()
    assert args.feat_set in FEATURE_SETS
    # Create the output directory if it does not exist
    os.makedirs(args.out_dir,exist_ok=True)
    # TODO: For now - read all the timed-unit files and extract features for all
    dialogue_names = [os.path.basename(p).split(".")[0] for p in glob.glob("{}/*.xml".format(TIMED_UNIT_PATHS))]
    run_pipeline(
        dialogue_names_split=dialogue_names,
        output_dir=args.out_dir,
        num_threads=5,
        feature_set=args.feat_set)








