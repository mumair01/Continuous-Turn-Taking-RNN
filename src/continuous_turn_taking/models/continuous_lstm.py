# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-26 18:35:51
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-08-28 10:36:48


from pickletools import floatnl
import pytorch_lightning as pl
import torch
import torch.nn as nn

class LSTMVoiceActivityPredictor(pl.LightningModule):

    _SUPPORTED_LOSS = ("mae",)

    def __init__(
        self,
        input_dim : int,
        hidden_dim : int,
        output_dim : int,
        layer_dim : int = 1,
        loss_fn : str = "mae",
        learning_rate : float = 0.01,
        weight_decay : float = 0.001
    ):

        assert loss_fn in self._SUPPORTED_LOSS, \
            f"ERROR: Invalid loss function {loss_fn}, must be one of: {self._SUPPORTED_LOSS}"

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_dim = layer_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Define the model
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layer_dim,
            batch_first=True
        )
        self.fc = nn.Linear(
            in_features=hidden_dim,
            out_features=output_dim,
            bias=True
        )
        self.fc_activation=nn.Sigmoid()

        # Generate the loss fn
        if loss_fn == "mae":
            self.loss_fn = nn.L1Loss()

    def forward(self, batch):
        """
        Args:
            batch: Shape (seq, batch, feature)
        """
        # Initialize the hidden states for first input with zeroes.
        h0 = torch.zeros(self.layer_dim,batch.size(0),self.hidden_dim).requires_grad_()
        # Initialize the cell state for the first input with zeroes.
        c0 = torch.zeros(self.layer_dim, batch.size(0),self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(batch.float(),(h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :] # Because we need the final output.
        return self.fc_activation(self.fc(out))

    def training_step(self, batch, batch_idx):
        """Calculate, log, and return the loss for a single training batch"""
        x, y = batch
        x = x.type(torch.FloatTensor)
        y_hat = self.forward(x)
        assert not torch.isnan(y_hat).any()
        assert not torch.isnan(y).any()
        loss = self.loss_fn(y, y_hat)
        assert not torch.isnan(y).any()
        self.log({"train/loss" : loss})
        return loss

    def validation_step(self, batch, batch_idx):
        """Calculate and log the loss for a single validation batch"""
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y, y_hat)
        self.log({"eval/loss" : loss})

    def configure_optimizers(self):
        return torch.optim.RMSprop(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )