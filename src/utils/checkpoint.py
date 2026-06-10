import torch

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_val_loss,
    word_to_idx,
    idx_to_word,
):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "word_to_idx": word_to_idx,
        "idx_to_word": idx_to_word,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return (
        checkpoint["epoch"],
        checkpoint["best_val_loss"],
        checkpoint["word_to_idx"],
        checkpoint["idx_to_word"],
    )