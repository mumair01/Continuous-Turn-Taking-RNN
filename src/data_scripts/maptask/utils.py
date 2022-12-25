# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-05-18 09:58:46
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-06-20 12:09:28

import os
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