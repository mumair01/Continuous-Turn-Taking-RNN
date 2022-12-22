# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-12-22 11:59:45
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-22 12:12:19

import pytest
import sys
import os

from models.va_predictor import VoiceActivityPredictor

def test_va_predictor():
    model = VoiceActivityPredictor(
        input_dim=130,
        hidden_dim=40, # Number of features of the hidden state
        out_features=30, # Number of output time steps,
        num_layers=1
    )
    print(model)