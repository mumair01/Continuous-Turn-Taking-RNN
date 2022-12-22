# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-05-18 09:58:42
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-21 13:11:38

import xml.etree.ElementTree
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
# Locals
from utils import *


# --------------------- FEATURE EXTRACTION METHODS --------------------------



# NOTE: This voice activity is for 50ms intervals.
def get_voice_activity_annotations(dialogue_name, participant):
    timed_unit_path = get_timed_unit(dialogue_name,participant)
    # Read the xml file
    tree = xml.etree.ElementTree.parse(timed_unit_path).getroot()
    # Extracting the audio end time from te timed units file.
    audio_end_time_ms = float(list(tree.iter())[-1].get('end')) *1000
    tu_tags = tree.findall('tu')
    # Getting all the times in which there are voice activity annotations in the corpus.
    va_times = []
    for tu_tag in tu_tags:
        start_time_s = float(tu_tag.get('start'))
        end_time_s = float(tu_tag.get('end'))
        if end_time_s - start_time_s >= MINIMUM_VA_CLASSIFICATION_TIME_MS/1000:
            va_times.append((start_time_s,end_time_s))
    # Get the frame times based on the final times unit time.
    # NOTE: This is being generated based on the step size for now.
    frame_times_s = np.arange(0,audio_end_time_ms,FRAME_STEP_MS) / 1000
    # Array to store voice  activity - initially all zeros means no voice activity.
    voice_activity = np.zeros((frame_times_s.shape[0]))
    # For each activity detected, get the start and end index of the nearest frame being
    # considered from the input audio.
    for start_time_s, end_time_s in va_times:
        # Obtaining index relative to the frameTimes being considered for
        # which there is voice activity.
        start_idx = np.abs(frame_times_s-start_time_s).argmin()
        end_idx = np.abs(frame_times_s-end_time_s).argmin()
        voice_activity[start_idx:end_idx+1] = VOICE_ACTIVITY_LABEL
    # Ensure that there are no nan values introduced in the data.
    assert not np.isnan(voice_activity).any() and not np.isnan(frame_times_s).any()
    return pd.DataFrame({
        "frameTime" : frame_times_s,
        "voiceActivity" :  voice_activity
    })

# Creating a method for extraction.

def extract_pitch(dialogue_name, participant):
    # Define the amount by which the opensmile features are shifted back
    # NOTE: This is because we are using a 50ms timestep and the extracted features
    # are also on a 50 ms timescale.

    # Read the gemaps feature file.
    gemaps_path = read_data(CORRECTED_GEMAPS_DIR,dialogue_name, participant,"csv")[0]
    gemaps_df = pd.read_csv(gemaps_path,delimiter=GEMAPS_CSV_DELIMITER,index_col=0)
    # Extract the relevant raw gemap features into a separate file.
    relevant_gemaps_df = gemaps_df[RELEVANT_GEMAP_FEATURES]
    # Obtain the z normalized values for each column individually.
    z_normalized_feat_df = relevant_gemaps_df.apply(
        lambda col: preprocessing.scale(col),axis=1,result_type='broadcast')
    # Make sure there are no nan values introduced
    assert not relevant_gemaps_df.isnull().values.any()
    assert not z_normalized_feat_df.isnull().values.any()
    # The original paper uses both the absolute and relevant pitch values and a
    # binary label indicating whether the frame was voiced.
    absolute_pitch = relevant_gemaps_df[PITCH_FEATURE_LABELS]
    z_normalized_pitch = z_normalized_feat_df[PITCH_FEATURE_LABELS]
    assert len(absolute_pitch) == len(z_normalized_pitch)
    # Determine whether frame was voiced
    frame_times_s = gemaps_df["frameTime"]
    data = {
        "frameTime" : frame_times_s,
        "{}Absolute".format(PITCH_FEATURE_LABELS) : absolute_pitch,
        "{}Znormalized".format(PITCH_FEATURE_LABELS) : z_normalized_pitch,
    }
    pitch_df = pd.DataFrame(data)
    assert not pitch_df.isnull().values.any()
    return pitch_df


def extract_power(dialogue_name, participant):
    # Define the amount by which the opensmile features are shifted back
    # NOTE: This is because we are using a 50ms timestep and the extracted features
    # are also on a 50 ms timescale.
    # Read the gemaps feature file.
    gemaps_path = read_data(CORRECTED_GEMAPS_DIR,dialogue_name, participant,"csv")[0]
    gemaps_df = pd.read_csv(gemaps_path,delimiter=GEMAPS_CSV_DELIMITER,index_col=0)
    # Extract the relevant raw gemap features into a separate file.
    relevant_gemaps_df = gemaps_df[RELEVANT_GEMAP_FEATURES]
    # Obtain the z normalized values for each column individually.
    z_normalized_feat_df = relevant_gemaps_df.apply(
        lambda col: preprocessing.scale(col),axis=1,result_type='broadcast')
    # Make sure there are no nan values introduced
    assert not relevant_gemaps_df.isnull().values.any()
    assert not z_normalized_feat_df.isnull().values.any()
    # The original paper uses power / intensity in dB -
    # TODO: Check what the units of loudness are in the GeMAPS set.
    absolute_power = relevant_gemaps_df[POWER_FEATURE_LABELS]
    z_normalized_power = z_normalized_feat_df[POWER_FEATURE_LABELS]
    # Determine whether frame was voiced
    frame_times_s = gemaps_df["frameTime"]
    data = {
        "frameTime" : frame_times_s,
        "{}_Absolute".format(POWER_FEATURE_LABELS) :  absolute_power,
        "{}_Znormalized".format(POWER_FEATURE_LABELS) : z_normalized_power
    }
    power_df = pd.DataFrame(data)
    assert not power_df.isnull().values.any()
    return power_df

def extract_spectral_flux(dialogue_name, participant):
    # Define the amount by which the opensmile features are shifted back
    # NOTE: This is because we are using a 50ms timestep and the extracted features
    # are also on a 50 ms timescale.
    # Read the gemaps feature file.
    gemaps_path = read_data(CORRECTED_GEMAPS_DIR,dialogue_name, participant,"csv")[0]
    gemaps_df = pd.read_csv(gemaps_path,delimiter=GEMAPS_CSV_DELIMITER)
    # Extract the relevant raw gemap features into a separate file.
    spectral_flux_df = gemaps_df[SPECTRAL_FEATURE_LABELS]
    # Obtain the z normalized values for each column individually.
    z_normalized_spectral_flux = preprocessing.scale(spectral_flux_df)
    # Make sure there are no nan values introduced
    assert not spectral_flux_df.isnull().values.any()
    assert not  np.isnan(z_normalized_spectral_flux).any()
    # Determine whether frame was voiced
    frame_times_s = gemaps_df["frameTime"]
    data = {
        "frameTime" : frame_times_s,
        "{}_Znormalized".format(SPECTRAL_FEATURE_LABELS) : z_normalized_spectral_flux
    }
    result_df = pd.DataFrame(data)
    assert not result_df.isnull().values.any()
    return result_df

# Combinging the above individual cells to create a single method to extract POS
# features

def extract_pos_annotations_with_delay(dialogue_name, participant):

    pos_tags_to_idx = {}
    idx_to_pos_tag = {}
    # NOTE: Indices start from 1 here because 0 already represents unknown categories.
    for i,tag in enumerate(POS_TAGS):
        pos_tags_to_idx[tag] = i +1
        idx_to_pos_tag[i+1] = tag

    # Need to read the timed-unit file for the corresponding start and end times
    # for the tags since we need to assign it
    timed_unit_path = read_data(
        TIMED_UNIT_PATHS,dialogue_name,participant,"xml")[0]
    tree_timed_unit = xml.etree.ElementTree.parse(timed_unit_path).getroot()
    tu_tags = tree_timed_unit.findall("tu")
    # Read the appropriate pos file
    pos_path = read_data(POS_PATH,dialogue_name,participant,"xml")[0]
    tree_pos = xml.etree.ElementTree.parse(pos_path).getroot()
    # The pos is the tag attribute in all the tw tags.
    tw_tags = tree_pos.findall("tw")
    # Getting all the times in which there are voice activity annotations in the corpus.
    va_times = []
    for tu_tag in tu_tags:
        start_time_s = float(tu_tag.get('start'))
        end_time_s = float(tu_tag.get('end'))
        if end_time_s - start_time_s >= MINIMUM_VA_CLASSIFICATION_TIME_MS/1000:
            va_times.append((start_time_s,end_time_s))
    # Extracting the audio end time from the timed units file.
    audio_end_time_ms = float(list(tree_timed_unit.iter())[-1].get('end')) *1000
    # Get the frame times based on the final times unit time.
    # NOTE: This is being generated based on the step size for now.
    frame_times_s = np.arange(0,audio_end_time_ms,FRAME_STEP_MS) / 1000
    # Collecting the end time of the word and the corresponding POS tag.
    word_annotations = []
    for tu_tag in tu_tags:
        tu_tag_id = tu_tag.get("id")[7:]
        end_time_s = float(tu_tag.get('end'))
        for tw_tag in tw_tags:
            # NOTE: Not sure if this is the correct way to extract the corresponding
            # timed-unit id.
            href = list(tw_tag.iter())[1].get("href")
            href_filename, href_ids = href.split("#")
            # Look at the appropriate file tags based on the filename.
            href_ids = href_ids.split("..")
            for href_id in href_ids:
                href_id = href_id[href_id.find("(")+8:href_id.rfind(")")]
                if href_id == tu_tag_id:
                    if tw_tag.get("tag") in POS_TAGS:
                        word_annotations.append((end_time_s,tw_tag.get("tag")))
    # For all the collected word end times and POS tags, we need to introduce
    # a delay and add the POS annotation to the delayed frame.
    pos_annotations = np.zeros((frame_times_s.shape[0]))
    for end_time_s, pos_tag in word_annotations:
        frame_idx = np.abs(frame_times_s-(end_time_s +POS_DELAY_TIME_MS/1000)).argmin()
        # Convert to integer based on the vocabulary dictionary.
        pos_annotations[frame_idx] = pos_tags_to_idx[pos_tag]
    # The pos annotations should not have any nan values
    assert not np.isnan(pos_annotations).any()
    # This encoder will ignore any unknown tags by replacing them with all zeros.
    onehot_encoder = OneHotEncoder(sparse=False,handle_unknown="ignore")
    onehot_encoder.fit(np.asarray(list(pos_tags_to_idx.values())).reshape(-1,1))
    encoded_pos = onehot_encoder.transform(pos_annotations.reshape(-1,1))
    pos_annotations_df = pd.DataFrame(encoded_pos,columns=POS_TAGS)
    # Add frametimes to the df
    pos_annotations_df.insert(0,"frameTime",frame_times_s)
    # Remove any duplicated frameTimes
    pos_annotations_df.drop_duplicates(subset=['frameTime'], inplace=True)
    assert not pos_annotations_df.isnull().values.any()
    return pos_annotations_df
