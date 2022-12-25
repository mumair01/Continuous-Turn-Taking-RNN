# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-05-18 09:58:24
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-06-20 13:58:07

import os

# All globals are defined here.

# -------------------- PATHS -------------------------


# Main paths
REPO_PATH = "/Users/muhammadumair/Documents/Repositories/mumair01-repos/TRP-Modeling/"
DATA_PATH = os.path.join(REPO_PATH,"skantze_2017_continuous/data/raw/maptask")
MAPTASK_PATH = os.path.join(DATA_PATH,"maptaskv2-1")
# NOTE: Using the 50ms GeMaps here.
GEMAPS_PATH = os.path.join(DATA_PATH,"audio_features/egemaps_v02_50ms")
RESULTS_PATH = os.path.join(REPO_PATH, "skantze_2017_continuous/data/processed","maptask")
assert os.path.isdir(REPO_PATH)
assert os.path.isdir(DATA_PATH)
assert os.path.isdir(MAPTASK_PATH)
assert os.path.isdir(REPO_PATH)
assert os.path.isdir(GEMAPS_PATH)

# Paths within the maptask corpus
STEREO_AUDIO_PATH = os.path.join(MAPTASK_PATH,"Data/signals/dialogues")
MONO_AUDIO_PATH = os.path.join(MAPTASK_PATH,"Data/signals/mono_signals")
# NOTE: The timed units are also used for Voice Activity annotations.
TIMED_UNIT_PATHS = os.path.join(MAPTASK_PATH,"Data/timed-units")
POS_PATH = os.path.join(MAPTASK_PATH,"Data/pos")

# -------------------- Extracted GeMAPs vars. -------------------------

GEMAPS_CSV_DELIMITER = ";"


# -------------------- MapTask vars. -------------------------

PARTICIPANT_LABELS_MAPTASK = ["f","g"] # NOTE: f = follower ; g = giver.



# -------------------- Pipeline params -------------------------

FRAME_STEP_MS = 50 # In the original paper, features were extracted every 50 ms.
FRAME_SIZE_MS = 50 # In the original paper, each frame was 50 ms long.


# ---- Voice Activity

# Minimum utterance duration for it to be considered voice activity.
MINIMUM_VA_CLASSIFICATION_TIME_MS = 25
VOICE_ACTIVITY_LABEL = 1 # This means that voice activity was detected.


# ---- Feature Definitions

# NOTE: These categories are defined in the original paper.
# NOTE: There are other features extracted by OpenSmile but they were not defined
# in the original paper - and are not used here - but these might be useful later.
# Ex. Mfccs.
# The names have been adapted for use with the results produced by opensmile.


GEMAPS_FREQUENCY_FEATURES = [
    'F0semitoneFrom27.5Hz_sma3nz', # Pitch: logarithmic F0 on a semitone frequency scale, starting at 27.5 Hz (semitone 0)
    "jitterLocal_sma3nz", # Jitter, deviations in individual consecutive F0 period lengths.
     # Formant 1, 2, and 3 frequency, centre frequency of first, second, and third formant
    "F1frequency_sma3nz",
    "F2frequency_sma3nz",
    "F3frequency_sma3nz",
    "F1bandwidth_sma3nz"
]

GEMAPS_ENERGY_FEATURES = [
    "shimmerLocaldB_sma3nz", # Shimmer, difference of the peak amplitudes of consecutive F0 periods.
    "Loudness_sma3", # Loudness, estimate of perceived signal intensity from an auditory spectrum.
    "HNRdBACF_sma3nz" # Harmonics-to-Noise Ratio (HNR), relation of energy in harmonic components to energy in noiselike components.
]

GEMAPS_SPECTRAL_FEATURES = [
    "alphaRatio_sma3", #  Alpha Ratio, ratio of the summed energy from 50–1000 Hz and 1–5 kHz
    "hammarbergIndex_sma3",  # Hammarberg Index, ratio of the strongest energy peak in the 0–2 kHz region to the strongest peak in the 2–5 kHz region
    # Spectral Slope 0–500 Hz and 500–1500 Hz, linear regression slope of the logarithmic power spectrum within the two given bands
    "slope0-500_sma3",
    "slope500-1500_sma3",
    # Formant 1, 2, and 3 relative energy, as well as the ratio of the energy of the spectral harmonic
    # peak at the first, second, third formant’s centre frequency to the energy of the spectral peak at F0.
    "F1amplitudeLogRelF0_sma3nz",
    "F2amplitudeLogRelF0_sma3nz",
    "F3amplitudeLogRelF0_sma3nz",
    "logRelF0-H1-H2_sma3nz", # Harmonic difference H1–H2, ratio of energy of the first F0 harmonic (H1) to the energy of the second F0 harmonic (H2)
    "logRelF0-H1-A3_sma3nz" # Harmonic difference H1–A3, ratio of energy of the first F0 harmonic (H1) to the energy of the highest harmonic in the third formant range (A3).
]

# These are all the GeMAPS features we are interested in.
RELEVANT_GEMAP_FEATURES = GEMAPS_FREQUENCY_FEATURES + GEMAPS_ENERGY_FEATURES + \
    GEMAPS_SPECTRAL_FEATURES

#  ---- Pitch
PITCH_FEATURE_LABELS = "F0semitoneFrom27.5Hz_sma3nz"

#  ---- Power / Intensity
POWER_FEATURE_LABELS = "Loudness_sma3"

# ---- Spectral

SPECTRAL_FEATURE_LABELS = 'spectralFlux_sma3'


# ---- Parts of Speech

POS_DELAY_TIME_MS = 100 # Assume that each POS calculation is delayed by 100ms.

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
    "ppg2\"",
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


