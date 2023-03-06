import math

import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel

from model.modelinterface import ModelInterface


class ConvolutionModel(ModelInterface):
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
        super().__init__(
            token_len, hidden_dim, out_dim, dropout_rate, kernel_size, stride, load_bert
        )
        self.token_len = token_len
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.config = RobertaConfig.from_pretrained("./model/config.json")
        self.bert = RobertaModel(config=self.config)
        if load_bert:
            self.bert.load_state_dict(torch.load("./model/bert_model.pth"))
        self.kernel_size = kernel_size
        self.stride = stride
        self.flatten = nn.Flatten()
        self.Conv1d1 = nn.Conv1d(
            self.hidden_dim,
            self.hidden_dim // 2,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )
        self.Conv1d2 = nn.Conv1d(
            self.hidden_dim // 2,
            self.hidden_dim // 4,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )
        self.ConvOut = (
            math.ceil(
                ((self.hidden_dim // 2 + 2 * 0 - (self.kernel_size - 1) - 1) + 1)
                / self.stride
            )
            + 1
        )
        self.batch_norm1 = nn.BatchNorm1d(self.hidden_dim * 4, affine=False)
        self.batch_norm2 = nn.BatchNorm1d(self.hidden_dim * 2, affine=False)
        self.ReLU = nn.ReLU()
        self.Linear1 = nn.Linear(24320, self.hidden_dim * 4)
        self.Linear2 = nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2)
        self.Linear3 = nn.Linear(self.hidden_dim * 2, out_dim)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.params = (
            list(self.Linear1.parameters())
            + list(self.Linear2.parameters())
            + list(self.Linear3.parameters())
            + list(self.Conv1d1.parameters())
            + list(self.Conv1d2.parameters())
        )

    def model(self, input):
        batch_size = input.size()[0]
        out = self.ReLU(self.Conv1d1(input))
        out = self.ReLU(self.Conv1d2(out))
        out = self.flatten(out)
        out = self.ReLU(self.Linear1(out))
        out = self.ReLU(self.Linear2(out))
        out = self.Linear3(out)
        return out

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask=attention_mask)
        out = out["last_hidden_state"]
        out = self.model(out)
        return out

    def predict(self, out, system_out, system_dicision, crowd_dicision, annotator):
        model_ans = []
        system_crowd = []
        s_count, c_count, a_count = 0, 0, 0
        # out = torch.stack((system_out, out[:, 1], out[:, 2]), -1)
        index = torch.argmax(out, dim=1)
        for i, idx in enumerate(index):
            if idx == 0:
                model_ans.append(system_dicision[i])
                s_count += 1
                system_crowd.append("system")
            elif idx == 1:
                model_ans.append(crowd_dicision[i])
                c_count += 1
                system_crowd.append("crowd")
            else:
                model_ans.append(annotator[i])
                a_count += 1
                system_crowd.append("annotator")
        model_ans = torch.Tensor(model_ans)
        return model_ans, s_count, c_count, a_count, system_crowd

    def get_params(self):
        return self.params
