from pathlib import Path
import random

import torch 
from torch.utils.data import Dataset
from PIL import Image

from src.configs.config import IMAGE_DIR
from src.data.text_preprocessing import encode_caption

class CaptionDataset(Dataset):
    def __init__(self,
        image_names,
        captions,
        word_to_idx,
        transform=None,
    ):        
        self.image_names = image_names
        self.captions = captions
        self.word_to_idx = word_to_idx
        self.transform = transform

        self.samples = []
        for image_name in self.image_names:
            if image_name in self.captions and len(self.captions[image_name]) > 0:
                self.samples.append(image_name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name = self.samples[idx]
        image_path = Path(IMAGE_DIR) / image_name
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        caption = random.choice(self.captions[image_name])
        caption = encode_caption(caption, self.word_to_idx)
        caption_tensor = torch.tensor(caption, dtype=torch.long)
        return image, caption_tensor