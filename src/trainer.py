import os
import torch
import pandas as pd
import matplotlib.pyplot as plt


def train_one_epoch(model,
                    train_loader,
                    criterion,
                    optimizer,
                    device,
                    scaler=None,
                    max_grad_norm=1.0):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None and device.startswith("cuda"):
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy


@torch.no_grad()
def validate_one_epoch(model,
                       val_loader,
                       criterion,
                       device):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if device.startswith("cuda"):
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy


def train_model(model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                device,
                epochs,
                output_dir,
                scheduler=None,
                patience=10,
                max_grad_norm=1.0):

    os.makedirs(output_dir, exist_ok=True)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = -1.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    scaler = torch.amp.GradScaler("cuda") if device.startswith("cuda") else None

    for epoch in range(epochs):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            max_grad_norm=max_grad_norm
        )

        val_loss, val_acc = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, "best_model.pth")
            )
        else:
            epochs_without_improvement += 1

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.2e}"
        )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs).")
            break

    print("\nTraining Complete.")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

    # If best_model.pth wasn't saved (e.g., edge case), save current model
    if not os.path.exists(os.path.join(output_dir, "best_model.pth")):
        torch.save(
            model.state_dict(),
            os.path.join(output_dir, "best_model.pth")
        )

    # ----------------------------
    # Save History CSV
    # ----------------------------

    history_df = pd.DataFrame(history)
    history_df.index += 1
    history_df.index.name = "Epoch"

    history_df.to_csv(
        os.path.join(output_dir, "history.csv")
    )

    # ----------------------------
    # Loss Curve
    # ----------------------------

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig(
        os.path.join(output_dir, "loss_curve.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # ----------------------------
    # Accuracy Curve
    # ----------------------------

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.savefig(
        os.path.join(output_dir, "accuracy_curve.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    return history