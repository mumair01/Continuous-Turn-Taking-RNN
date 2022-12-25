# -*- coding: utf-8 -*-
# @Author: Muhammad Umair
# @Date:   2022-08-26 18:35:51
# @Last Modified by:   Muhammad Umair
# @Last Modified time: 2022-12-24 20:08:08


import pytorch_lightning as pl
import torch
import torch.nn as nn

class VoiceActivityPredictor(pl.LightningModule):

    def __init__(
        self,
        input_dim : int,
        hidden_dim : int,
        out_features : int,
        num_layers : int = 1,
        loss_fn : str = "mae",
        learning_rate : float = 0.01,
        weight_decay : float = 0.001
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.num_layers = num_layers
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Define the layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(
                in_features=hidden_dim,
                out_features=out_features
            ),
            nn.Sigmoid()
        )

        if loss_fn == "mae":
            self.loss_fn = nn.L1Loss()
        else:
            raise NotImplementedError(
                f"Loss not implemented: {loss_fn}"
            )

    def forward(self, batch):
        """
        Args:
            batch: Tensor of shape: (batch_size, seq_len, feature)
        """
        # Initialize the hidden states for first input with zeroes.
        h0 = torch.zeros(
            self.num_layers,batch.size(0),self.hidden_dim
        ).requires_grad_()
        # Initialize the cell state for the first input with zeroes.
        c0 = torch.zeros(
            self.num_layers, batch.size(0),self.hidden_dim
        ).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through
        # time (BPTT). If we don't, we'll back propagate all the way to the
        # start even after going through another batch.
        # Forward propagation by passing in the input, hidden state,
        # and cell state into the model
        # out is the output of the RNN from all timesteps from the last RNN layer.
        # h_n /c_n is the hidden value from the last time-step of all RNN layers.
        # out shape: (batch_size, seq_len, 1 * hidden_size)
        # h_n shape: (num_layers * 1, batch_size, hidden_size)
        out, _ = self.lstm(batch.float(),(h0.detach(), c0.detach()))

        # Based on the paper, we are interested in the prediction made at the
        # last timestamp.
        out = out[:,-1,:]
        # Fully connected is: Sigmoid(wx +b)
        out = self.fc(out)
        # out shape: (batch_size, output_dim)
        return out

    def training_step(self, batch, batch_idx):
        """
        Calculate, log, and return the loss for a single training batch.
        Args:
            batch splits into x,y.
                x shape: (batch_size, seq_len, features)
                y shape: (batch_size, N)
        """
        x, y = batch
        # x.type(torch.FloatTensor) # NOTE: Should remove
        out = self.forward(x) # out shape: (batch_size, N)
        loss = self.loss_fn(y, out)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Calculate and log the loss for a single validation batch.
        """
        x, y = batch
        out = self.forward(x)
        loss = self.loss_fn(y, out)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        """
        Calculate and log the loss for a single validation batch.
        """
        x, y = batch
        out = self.forward(x)
        loss = self.loss_fn(y, out)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        # This is the same optimizer used in the paper.
        return torch.optim.RMSprop(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

