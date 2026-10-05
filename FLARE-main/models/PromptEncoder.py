

import torch
import torch.nn as nn

# Adapted from: https://github.com/THUDM/P-tuning/blob/main/LAMA/p_tuning/prompt_encoder.py
# Copyright (c) 2021 THUDM
# License: MIT
class PromptEncoder(torch.nn.Module):
    def __init__(self, template, hidden_size, device, lstm_dropout):
        super().__init__()
        self.device = device
        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        # ent embedding
        self.lstm_dropout=lstm_dropout
        self.cloze_length = template
        self.cloze_mask = [
            [1] * self.cloze_length[0]  # first cloze
            + [1] * self.cloze_length[1]  # second cloze
            + [1] * self.cloze_length[2]  # third cloze
            + [1] * self.cloze_length[3]  # forth cloze
        ]
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool().to(self.device)

        self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))).to(self.device)
        # embedding
        self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.hidden_size).to(self.device)
        # LSTM
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))
        print("init prompt encoder...")

    def forward(self):
        input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0]).squeeze()
        return output_embeds
    
