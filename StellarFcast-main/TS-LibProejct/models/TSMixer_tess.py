import torch.nn as nn


class TSMixerBlock(nn.Module):
    def __init__(self, seq_len, enc_in, hidden_dim, dropout):
        super().__init__()
        self.time_mix = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len),
            nn.Dropout(dropout)
        )
        self.channel_mix = nn.Sequential(
            nn.Linear(enc_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, enc_in),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.time_mix(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel_mix(x)
        return x


class TSMixerForClassification:
    """
    Wrapper class to provide .Model interface compatible with Exp framework.
    """

    class Model(nn.Module):
        def __init__(self, args):
            super().__init__()
            seq_len = args.seq_len
            in_channels = args.enc_in
            num_classes = getattr(args, 'num_class', 2)
            hidden_dim = getattr(args, 'd_model', 128)  # 可复用 d_model
            num_layers = getattr(args, 'e_layers', 4)  # 可复用 e_layers
            dropout = getattr(args, 'dropout', 0.1)

            # Embedding layer: (B, T, C) -> (B, T, D)
            self.embedding = nn.Linear(in_channels, hidden_dim)

            # Stack TSMixer blocks
            self.blocks = nn.Sequential(*[
                TSMixerBlock(
                    seq_len=seq_len,
                    enc_in=hidden_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout
                ) for _ in range(num_layers)
            ])

            # Classifier
            self.classifier = nn.Linear(hidden_dim, num_classes)

        def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
            # x: (B, T, C)
            x = self.embedding(x)  # (B, T, D)
            x = self.blocks(x)  # (B, T, D)
            x = x.mean(dim=1)  # (B, D) — global average over time
            logits = self.classifier(x)  # (B, num_classes)
            return logits