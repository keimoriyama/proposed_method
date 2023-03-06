import math

import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel


class ModelInterface(nn.Module):
    def __init__(
        self,
        token_len,
        hidden_dim,
        out_dim,
        dropout_rate,
        kernel_size,
        stride,
        load_bert=False,
    ) -> None:
        super().__init__()

    def forward(self):
        raise NotImplementedError

    def model(self, input):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def get_params(self):
        raise NotImplementedError
