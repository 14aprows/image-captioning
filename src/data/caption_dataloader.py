import torch 
from torch.utils.data import DataLoader, random_split
from src.configs.config import (
   BATCH_SIZE,
   NUM_WORKERS,
   PIN_MEMORY,
   PAD_TOKEN,
   VAL_RATIO,
   RANDOM_SEED
)

from src.data.text_preprocessing import get_caption_data
from src.data.image_preprocessing import get_train_transform, get_eval_transform
from src.data.caption_dataset import CaptionDataset
from src.utils.collate_fn import caption_collate_fn

def get_caption_dataloader():
    captions, train_images, test_images, word_to_idx, idx_to_word = get_caption_data()
    pad_idx = word_to_idx[PAD_TOKEN]

    full_train_dataset = CaptionDataset(
        image_names=train_images,
        captions=captions,
        word_to_idx=word_to_idx,
        transform=get_train_transform(),
    )

    test_dataset = CaptionDataset(
        image_names=test_images,
        captions=captions,
        word_to_idx=word_to_idx,
        transform=get_eval_transform(),
    )

    val_size = int(len(full_train_dataset) * VAL_RATIO)
    train_size = len(full_train_dataset) - val_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size], 
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=lambda batch: caption_collate_fn(batch, pad_idx),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=lambda batch: caption_collate_fn(batch, pad_idx),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=lambda batch: caption_collate_fn(batch, pad_idx),
    )

    return train_loader, val_loader, test_loader, word_to_idx, idx_to_word