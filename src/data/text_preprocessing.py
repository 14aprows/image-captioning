import re
from collections import Counter
from src.configs.config import (
    END_TOKEN,
    MIN_WORD_FREQ,
    PAD_TOKEN,
    START_TOKEN,
    TOKEN_FILE,
    UNK_TOKEN,
    TEST_IMAGE_FILE,
    TRAIN_IMAGE_FILE,
)


def clean_caption(caption):
    caption = caption.lower()
    caption = re.sub(r"[^a-z0-9\s]", "", caption)
    caption = re.sub(r"\s+", " ", caption).strip()
    return caption


def tokenize_caption(caption):
    caption = clean_caption(caption)
    tokens = caption.split()
    return tokens


def read_image_list(file_path):
    image_names = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            image_name = line.strip()
            if image_name:
                image_names.append(image_name)
    return image_names


def read_caption_file(token_file):
    captions = {}

    with open(token_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue

            image_id, caption = parts
            image_name = image_id.split("#")[0]
            caption = clean_caption(caption)

            if image_name not in captions:
                captions[image_name] = []

            captions[image_name].append(caption)

    return captions


def build_vocab(captions, image_names):
    counter = Counter()
    for image_name in image_names:
        image_captions = captions.get(image_name, [])
        for caption in image_captions:
            tokens = tokenize_caption(caption)
            counter.update(tokens)

    word_to_idx = {
        PAD_TOKEN: 0,
        START_TOKEN: 1,
        END_TOKEN: 2,
        UNK_TOKEN: 3,
    }

    for word, freq in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        if freq >= MIN_WORD_FREQ:
            word_to_idx[word] = len(word_to_idx)

    idx_to_word = {v: k for k, v in word_to_idx.items()}
    return word_to_idx, idx_to_word


def encode_caption(caption, word_to_idx):
    tokens = tokenize_caption(caption)
    encoded = [word_to_idx[START_TOKEN]]
    for token in tokens:
        token_id = word_to_idx.get(token, word_to_idx[UNK_TOKEN])
        encoded.append(token_id)
    encoded.append(word_to_idx[END_TOKEN])
    return encoded


def get_caption_data():
    captions = read_caption_file(TOKEN_FILE)
    train_images = read_image_list(TRAIN_IMAGE_FILE)
    test_images = read_image_list(TEST_IMAGE_FILE)

    word_to_idx, idx_to_word = build_vocab(
        captions=captions,
        image_names=train_images
    )

    return captions, train_images, test_images, word_to_idx, idx_to_word
