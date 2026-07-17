import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# Create multilabel target matrix for stratified cross-validation
def create_multilabel_targets(dataset, tasks):
    y_multilabel = np.array([
        [
            targets[task]
            for task in tasks
        ]
        for x, targets in dataset
    ])

    return y_multilabel

# Create the multilabel stratified K-fold splitter
def create_multilabel_splitter(n_splits=10, random_state=522):
    mskf = MultilabelStratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    return mskf