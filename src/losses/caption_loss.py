import torch 
import torch.nn as nn

class CaptionLoss(nn.Module):
    def __init__(self, pad_idx):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_idx
        )

    def forward(self, logits, captions):
        logits = logits[:, :-1, :]
        vocab_size = logits.size(-1)
        logits = logits.reshape(-1, vocab_size)
        targets = captions.reshape(-1)
        loss = self.criterion(logits, targets)
        return loss
        
