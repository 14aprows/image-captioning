import torch 
import torch.nn as nn

class LSTMDecoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        vocab_size,
        num_layers,
        dropout
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        features = features.unsqueeze(1)
        lstm_input = torch.cat((features, embeddings), dim=1)
        lstm_input = self.dropout(lstm_input)
        lstm_output, _ = self.lstm(lstm_input)
        logits = self.fc(lstm_output)
        return logits
