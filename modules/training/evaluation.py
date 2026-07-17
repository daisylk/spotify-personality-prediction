import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score, mean_absolute_error, confusion_matrix, classification_report)

from training.losses import get_prediction

# Model evaluation across all prediction taks
def evaluate_model(model, test_loader, task_info, tasks):
    results = {
        task: {"true": [], "pred": []}
        for task in tasks
    }

    model.eval()

    with torch.no_grad():
        for X_user, targets in test_loader:
            outputs = model(X_user)

            for task in outputs:

                task_logits = outputs[task].squeeze(0)

                pred = get_prediction(
                    task,
                    task_logits,
                    task_info
                )

                results[task]["true"].append(
                    int(targets[task].item())
                )

                results[task]["pred"].append(
                    int(pred)
                )

    evaluation_results = {}

    for task in results:
        y_true_np = np.array(results[task]["true"])
        y_pred_np = np.array(results[task]["pred"])

        acc = accuracy_score(y_true_np, y_pred_np)
        f1 = f1_score(y_true_np, y_pred_np, average="macro")
        bal_acc = balanced_accuracy_score(y_true_np, y_pred_np)

        # MAE is only calculated for the ordinal tasks
        if task_info[task]["type"] == "ordinal":
            mae = mean_absolute_error(y_true_np, y_pred_np)
        else:
            mae = None

        # Store confusion matrices and classification reports
        cm = confusion_matrix(y_true_np, y_pred_np, labels=range(task_info[task]["n_classes"]))
        report = classification_report(y_true_np, y_pred_np, output_dict=True, zero_division=0)

        evaluation_results[task] = {
            "accuracy": acc,
            "f1": f1,
            "balanced_accuracy": bal_acc,
            "mae": mae,
            "confusion_matrix": cm,
            "report": report
        }

    return evaluation_results