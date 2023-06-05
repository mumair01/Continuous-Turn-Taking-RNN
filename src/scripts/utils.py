# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2023-06-05 10:19:44
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-06-05 10:21:36


import toml
from typing import Dict

_CONFIG_PATH = "./conf.toml"


def load_user_configs() -> Dict:
    try:
        return toml.load(_CONFIG_PATH)
    except:
        raise Exception(f"User configuration load failed from: {_CONFIG_PATH}")
