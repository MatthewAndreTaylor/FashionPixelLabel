from tqdm import tqdm
import torch
import torch.nn as nn


def fit(
    epochs: int,
    optimizer: torch.optim,
    model: nn.Module,
    loss_fn: nn.Module,
    learning_rate: float,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    learning_rate_scheduler: torch.optim.lr_scheduler,
    **kwargs,
):
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    lrs = learning_rate_scheduler(optimizer, **kwargs)
    min_val_loss = 1
    old_lr = learning_rate

    for epoch in range(epochs):
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"
        )

        for batch in progress_bar:
            optimizer.zero_grad()
            model.train()
            index, image, label = batch
            prediction = model(image)
            loss = loss_fn(prediction, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            progress_bar.set_postfix(train_loss=loss.item())

        val_losses = []
        for batch in val_loader:
            model.eval()
            with torch.no_grad():
                index, image, label = batch
                prediction = model(image)
                val_loss = loss_fn(prediction, label)
                val_losses.append(val_loss)

        epoch_loss = torch.stack(val_losses).mean().item()

        # Print the validation loss
        print(f"Epoch {epoch + 1}/{epochs}, val_loss: {epoch_loss}")

        lrs.step(epoch_loss)

        if optimizer.param_groups[0]["lr"] != old_lr:
            old_lr = optimizer.param_groups[0]["lr"]

        if epoch_loss < min_val_loss:
            torch.save(
                model, f"saved_models/{type(model).__name__}_{model.num_classes}.pt"
            )
            min_val_loss = epoch_loss
