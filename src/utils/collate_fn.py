import torch 
from torch.nn.utils.rnn import pad_sequence

def caption_collate_fn(batch, pad_idx):
    images = []
    captions = []

    for image, caption in batch:
        images.append(image)
        captions.append(caption)

    images = torch.stack(images, dim=0)
    captions = pad_sequence(
        captions,
        batch_first=True,
        padding_value=pad_idx
    )
    return images, captions