import torch 
from tqdm import tqdm

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    grad_clip
):
    model.train()

    total_loss = 0.0
    total_batches = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for images, captions in progress_bar:
        images = images.to(device)
        captions = captions.to(device)

        optimizer.zero_grad()
        logits = model(images, captions)
        loss = criterion(logits, captions)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        progress_bar.set_postfix(loss=total_loss / total_batches)
    avg_loss = total_loss / total_batches
    return avg_loss

def validate_one_epoch(
    model,
    dataloader,
    criterion,
    device
):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    progress_bar = tqdm(dataloader, desc="Validation", leave=False)

    with torch.no_grad():
        for images, captions in progress_bar:
            images = images.to(device)
            captions = captions.to(device)

            logits = model(images, captions)
            loss = criterion(logits, captions)

            total_loss += loss.item()
            total_batches += 1

            progress_bar.set_postfix(loss=total_loss / total_batches)
    avg_loss = total_loss / total_batches
    return avg_loss