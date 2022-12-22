
# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-05-18 09:58:46
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-21 13:15:18

import os
import pandas as pd
import numpy as np
from vardefs import *


def get_timed_unit(dialogue_name,participant):
   return read_data(TIMED_UNIT_PATHS,dialogue_name,participant,"xml")[0]

def get_mono_audio(dialogue_name, participant):
    return read_data(MONO_AUDIO_PATH,dialogue_name,participant,"wav")[0]

def get_stereo_audio(dialogue_name):
    return read_data(STEREO_AUDIO_PATH,dialogue_name,"mix","wav")[0]

def read_data(dir_path,dialogue_name, participant,ext):
    """
    Read files from a directory and return files with a specific dialogue
    and participant based on the maptask filename convention
    i.e., dialogue.participant...
    """
    results = []
    data_paths = [p for p in os.listdir(dir_path)]
    data_paths = [os.path.join(dir_path,p) for p in data_paths if os.path.splitext(p)[1][1:] == ext]
    for path in data_paths:
        basename = os.path.basename(path)
        filename ,file_ext = os.path.splitext(basename)
        split_filename = filename.split('.')
        file_dialogue = split_filename[0]
        file_participant = split_filename[1]
        if file_dialogue == dialogue_name and\
                file_participant == participant and \
                file_ext[1:] == ext:
            results.append(path)
    return results


# Utility function for reading GeMaps since the delimiter may be either , or ;
# TODO: For some reason q1ec1 is using , as delimiter instead of ;
def read_gemaps_as_df(path):
    for delimiter in (",",";"):
        try:
            gemaps_df = pd.read_csv(path,delimiter=delimiter,index_col=False)
            x = gemaps_df["frameTime"]
            return gemaps_df
        except:
            pass

def verify_correct_gemaps(gemaps_dir, dialogue_name, participant,result_dir):
    """
    Verify that the gemaps data does not have any duplicated frame times and
    that there are no null values.
    """
    gemaps_paths = read_data(gemaps_dir,dialogue_name, participant,"csv")
    for path in gemaps_paths:
        filename = os.path.splitext(os.path.basename(path))[0]
        gemaps_df = read_gemaps_as_df(path)
        # NOTE: This is required because opensmile is producing some duplicated frameTimes.
        gemaps_df.drop_duplicates(subset=['frameTime'], inplace=True)
        # Drop the 'name' column
        gemaps_df.drop(columns=['name'],inplace=True)
        # Check that the frameTime steps are as expected
        for i in range(len(gemaps_df['frameTime']) -1):
            difference = np.abs(gemaps_df['frameTime'].iloc[i+1] - gemaps_df['frameTime'].iloc[i])
            assert (FRAME_STEP_MS/1000 - 1e-3) < difference < (FRAME_STEP_MS/1000 + 1e-3)
        # Ensure that none of the values is null.
        assert not gemaps_df.isnull().values.any()
        gemaps_df.to_csv(os.path.join(result_dir,"{}.csv".format(filename)), sep=GEMAPS_CSV_DELIMITER)