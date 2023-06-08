# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2023-05-26 16:27:32
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-05-31 10:24:29

""" 
Contains various global literal definitions required by the maptask dataset 

"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MapTaskConstants:
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
        "sent",
    ]

    GEMAPS_FREQUENCY_FEATURES = [
        "F0semitoneFrom27.5Hz_sma3nz",  # Pitch: logarithmic F0 on a semitone frequency scale, starting at 27.5 Hz (semitone 0)
        # "jitterLocal_sma3nz", # Jitter, deviations in individual consecutive F0 period lengths.
        #  # Formant 1, 2, and 3 frequency, centre frequency of first, second, and third formant
        # "F1frequency_sma3nz",
        # "F2frequency_sma3nz",
        # "F3frequency_sma3nz",
        # "F1bandwidth_sma3nz"
    ]

    GEMAPS_ENERGY_FEATURES = [
        # "shimmerLocaldB_sma3nz", # Shimmer, difference of the peak amplitudes of consecutive F0 periods.
        "Loudness_sma3",  # Loudness, estimate of perceived signal intensity from an auditory spectrum.
        # "HNRdBACF_sma3nz" # Harmonics-to-Noise Ratio (HNR), relation of energy in harmonic components to energy in noiselike components.
    ]

    GEMAPS_SPECTRAL_FEATURES = [
        "spectralFlux_sma3",
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

    # Features for both the prosody model.
    # TODO: Check whether frameTime is a feature used in the paper.
    PROSODY_FEATURES = (
        GEMAPS_FREQUENCY_FEATURES
        + GEMAPS_ENERGY_FEATURES
        + GEMAPS_SPECTRAL_FEATURES
    )
    # Features for the Full model.
    FULL_FEATURES = PROSODY_FEATURES + POS_TAGS
