import torch 
import torch.nn as nn

from src.models.cnn_encoder import CNNEncoder
from src.models.lstm_decoder import LSTMDecoder

class ImageCaptioningModel(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        vocab_size,
        num_layers,
        dropout
    ):
        super().__init__()
        self.cnn_encoder = CNNEncoder(
            embed_dim=embed_dim,
        )
        self.lstm_decoder = LSTMDecoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            num_layers=num_layers,
            dropout=dropout
        )
    def forward(
        self,
        images,
        captions
    ):
        features = self.cnn_encoder(images)
        logits = self.lstm_decoder(features, captions)
        return logits