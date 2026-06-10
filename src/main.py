import torch
import torch.optim as optim

from src.configs.config import (
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    GRAD_CLIP,
    EMBED_DIM,
    HIDDEN_DIM,
    NUM_LAYERS,
    DROPOUT,
    PAD_TOKEN,
    CHECKPOINT_DIR,
)
from src.data.caption_dataloader import get_caption_dataloader
from src.models.captioning_model import ImageCaptioningModel
from src.losses.caption_loss import CaptionLoss
from src.engine.trainer import train_one_epoch, validate_one_epoch
from src.utils.checkpoint import save_checkpoint

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, word_to_idx, idx_to_word = (
        get_caption_dataloader()
    )
    vocab_size = len(word_to_idx)
    pad_idx = word_to_idx[PAD_TOKEN]

    model = ImageCaptioningModel(
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        vocab_size=vocab_size,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    criterion = CaptionLoss(pad_idx=pad_idx)
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    best_model_path = CHECKPOINT_DIR / f"{model.__class__.__name__}_best.pth"

    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            GRAD_CLIP,
        )
        val_loss = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
        )
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                path=best_model_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                word_to_idx=word_to_idx,
                idx_to_word=idx_to_word,
            )
            print(f"Checkpoint saved to {best_model_path}")

if __name__ == "__main__":
    main()