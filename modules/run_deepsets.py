import numpy as np
import pandas as pd
import torch
import os
import pickle
from torch.utils.data import DataLoader
from data.preprocessing import prepare_dataset, scale_dataset
from task_info import task_info, tasks
from training.cross_validation import (create_multilabel_targets, create_multilabel_splitter)
from data.dataset import SpotifyDataset
from training.trainer import train_model
from training.evaluation import evaluate_model
from models.deep_sets import DeepSetsModel


# Full run of the Deep Sets model
dataset = prepare_dataset()

X = np.array([x for x, targets in dataset], dtype=object)

y_multilabel = create_multilabel_targets(dataset, tasks)

mskf = create_multilabel_splitter()

fold_results = []

all_training_history = []

all_cm = {task: [] for task in tasks}
all_reports = {task: [] for task in tasks}

for fold, (train_idx, test_idx) in enumerate(
    mskf.split(X, y_multilabel)):

    print(f"\nFold { fold+1}")

    # Create outer test-train split

    fold_train_data = [
        dataset[i]
        for i in train_idx
    ]

    test_data = [
        dataset[i]
        for i in test_idx
    ]

    # Create validation split with training data

    X_inner = np.array(
        [x for x, y in fold_train_data],
        dtype=object
    )

    y_inner = create_multilabel_targets(
        fold_train_data,
        tasks
    )

    inner_splitter = create_multilabel_splitter(
        n_splits=5
    )

    inner_train_idx, val_idx = next(
        inner_splitter.split(
            X_inner,
            y_inner
        )
    )

    train_data = [
        fold_train_data[i]
        for i in inner_train_idx
    ]

    val_data = [
        fold_train_data[i]
        for i in val_idx
    ]

    # Calculate class weights from training data

    class_weights = {}

    for task, info in task_info.items():
        train_labels = np.array([
            targets[task]
            for x, targets in train_data
        ])

        class_counts = np.bincount(
            train_labels.astype(int),
            minlength=info["n_classes"]
        )

        task_weights = len(train_labels) / (
            len(class_counts) * np.maximum(class_counts, 1)
        )

        class_weights[task] = torch.tensor(
            task_weights,
            dtype=torch.float32
        )

    train_data, val_data, test_data = scale_dataset(
        train_data,
        val_data,
        test_data
    )

    # Create PyTorch datasets and data loaders

    train_dataset = SpotifyDataset(train_data)
    val_dataset = SpotifyDataset(val_data)
    test_dataset = SpotifyDataset(test_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )

    # Initialise model and optimiser

    model = DeepSetsModel(
        task_info
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0005,
        weight_decay=5e-4
    )

    # Train model

    training_results = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        task_info,
        class_weights,
        tasks
    )

    all_training_history.append(
        training_results
    )

    # Load model checkpoint with the lowest validation loss

    model.load_state_dict(
        training_results["best_model_state"]
    )

    # Evaluate on test fold
    
    evaluation_results = evaluate_model(
        model,
        test_loader,
        task_info,
        tasks
    )

    fold_results.append(evaluation_results)

    for task in tasks:

        all_cm[task].append(
            evaluation_results[task]["confusion_matrix"]
        )

        all_reports[task].append(
            evaluation_results[task]["report"]
        )

# Calculate mean cross-validation performance

for task in tasks:

    accuracies = [
        fold[task]["accuracy"]
        for fold in fold_results
    ]

    f1s = [
        fold[task]["f1"]
        for fold in fold_results
    ]

    bal_accs = [
        fold[task]["balanced_accuracy"]
        for fold in fold_results
    ]

    print(f"\n===== {task.upper()} =====")

    print(
        "Mean Accuracy:",
        f"{np.mean(accuracies):.3f} +/- {np.std(accuracies):.3f}"
    )

    print(
        "Mean Macro F1:",
        f"{np.mean(f1s):.3f} +/- {np.std(f1s):.3f}"
    )

    print(
        "Mean Balanced Accuracy:",
        f"{np.mean(bal_accs):.3f} +/- {np.std(bal_accs):.3f}"
    )

    if task_info[task]["type"] == "ordinal":
        
        maes = [
            fold[task]["mae"]
            for fold in fold_results
        ]

        print(
            "Mean MAE:",
            f"{np.mean(maes):.3f} +/- {np.std(maes):.3f}"
        )

# Aggregate confusion matrices across cross-validation folds

for task in tasks:

    final_cm = np.sum(
        all_cm[task],
        axis=0
    )

    print(f"\n{task} confusion matrix:")
    print(final_cm)


# Aggregate classification reports across cross-validation folds

for task in tasks:

    flat_reports = []

    for report in all_reports[task]:

        flat = {}

        for k, v in report.items():

            if isinstance(v, dict):

                for metric, score in v.items():
                    flat[f"{k}_{metric}"] = score

            else:
                flat[k] = v

        flat_reports.append(flat)

    report_df = pd.DataFrame(flat_reports)

    print(f"\nMean Report Metrics - {task}")
    print(
        report_df.mean(numeric_only=True)
    )

results = {
    "fold_results": fold_results,
    "confusion_matrices": all_cm,
    "classification_reports": all_reports,
    "training_history": all_training_history,
    "task_info": task_info
}

os.makedirs("results", exist_ok=True)

with open("results/deepsets_results_loss.pkl", "wb") as f:
    pickle.dump(results, f)