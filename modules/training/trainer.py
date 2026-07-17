import numpy as np
import torch
import copy
from sklearn.metrics import f1_score, balanced_accuracy_score
from training.losses import get_task_loss, get_prediction

# Multitask training functions for the Deep Sets and Multiple Instance Learning Models
# Train model for one epoch
def train_epoch(model, train_loader, optimizer, task_info, class_weights):
    model.train()

    epoch_loss = 0

    for X_user, targets in train_loader:
        optimizer.zero_grad()

        outputs = model(X_user)
        loss = 0

        for task in outputs:
            task_logits = outputs[task].squeeze(0)

            task_target = targets[task]

            task_loss = get_task_loss(
                task,
                task_logits,
                task_target,
                task_info,
                class_weights=class_weights[task]
            )

            loss += task_loss

        loss = loss / len(outputs) # Averages loss across task-specific outputs

        loss.backward()

        optimizer.step()
            
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)

    return avg_loss

# Evaluate performance on validation set
def validate_epoch(model, val_loader, task_info, class_weights, tasks):
    model.eval()

    val_loss = 0

    val_results = {
        task: {"true": [], "pred": []}
        for task in tasks
    }

    with torch.no_grad():
        for X_user, targets in val_loader:

            outputs = model(X_user)

            user_loss = 0

            for task in outputs:
                task_logits = outputs[task].squeeze(0)

                task_target = targets[task]

                task_loss = get_task_loss(
                    task,
                    task_logits,
                    task_target,
                    task_info,
                    class_weights=class_weights[task]
                )

                pred = get_prediction(
                    task,
                    task_logits,
                    task_info
                )

                # Compute macro f1 and balanced accuracy to compare across epochs
                val_results[task]["true"].append(
                    int(targets[task].item())
                )

                val_results[task]["pred"].append(
                    int(pred)
                )

                user_loss += task_loss

            # Average loss across the tasks
            user_loss = user_loss / len(outputs)

            val_loss += user_loss.item()

    avg_val_loss = val_loss / len(val_loader)

    task_f1s = []
    task_bal_accs = []

    for task in tasks:
        y_true = np.array(val_results[task]["true"])
        y_pred = np.array(val_results[task]["pred"])

        task_f1s.append(f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ))

        task_bal_accs.append(balanced_accuracy_score(
            y_true,
            y_pred
        ))

    avg_val_f1 = np.mean(task_f1s)
    avg_val_bal_acc = np.mean(task_bal_accs)

    return avg_val_loss, avg_val_f1, avg_val_bal_acc, val_results

# Train model with validation monitoring and early stopping
def train_model(model, train_loader, val_loader, optimizer, task_info, class_weights, tasks, epochs=50, patience=10):
    train_losses = []
    val_losses = []
    val_f1_history = []
    val_bal_acc_history = []

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    best_model_state = copy.deepcopy(
        model.state_dict()
    )

    for epoch in range(epochs):
        train_loss = train_epoch(
            model, 
            train_loader,
            optimizer,
            task_info,
            class_weights
        )

        val_loss, val_f1, val_bal_acc, val_results = validate_epoch(
            model,
            val_loader,
            task_info,
            class_weights,
            tasks
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1_history.append(val_f1)
        val_bal_acc_history.append(val_bal_acc)

        print(
            f"Epoch {epoch+1}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val F1: {val_f1:.4f}, "
            f"Val Bal Acc: {val_bal_acc:.4f}"
        )

        # Save model state when validation loss improves
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            
            best_model_state = copy.deepcopy(
                model.state_dict()
            )

        else:
            epochs_without_improvement += 1

        # Stop training when validation loss does not improve for the patience period
        if epochs_without_improvement >= patience:

            print(
                f"Early stopping at epoch {epoch+1}"
            )

            break

    print(
        f"Best validation loss: {best_val_loss:.4f} "
        f"(epoch {best_epoch})"
    )

    best_f1_epoch = np.argmax(val_f1_history) + 1
    best_bal_epoch = np.argmax(val_bal_acc_history) + 1

    print(
        f"Best validation F1: {max(val_f1_history):.4f} "
        f"(epoch {best_f1_epoch})"
    )

    print(
        f"Best validation bal acc: {max(val_bal_acc_history):.4f} "
        f"(epoch {best_bal_epoch})"
    )

    return {
        "best_model_state": best_model_state,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_f1_history": val_f1_history,
        "val_bal_acc_history": val_bal_acc_history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss
    }