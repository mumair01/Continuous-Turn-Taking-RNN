# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2023-06-04 13:50:56
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2023-06-04 14:36:38

import pytest
from turn_taking.models.gcttLSTM import GCTTLSTM
from turn_taking.dsets.maptask import MapTaskVADataModule


from pytorch_lightning import Trainer


def test_va_predictor(
    result_dir,
    force_reprocess,
):
    dm = MapTaskVADataModule(
        data_dir=result_dir,
        sequence_length_ms=10_000,
        prediction_length_ms=1000,
        feature_set="full",
        target_participant="f",
        batch_size=1,
        num_conversations=1,
        force_reprocess=force_reprocess,
    )
    dm.prepare_data()
    dm.setup(stage="fit")
    # Checking the dims.
    loader = dm.train_dataloader()
    x, y = next(iter(loader))
    print(x.shape, y.shape)

    model = GCTTLSTM(
        input_dim=130,
        hidden_dim=40,  # Number of features of the hidden state
        out_features=100,  # Number of output time steps,
        num_layers=1,
    )

    trainer = Trainer(max_epochs=1)
    trainer.fit(model, dm)
