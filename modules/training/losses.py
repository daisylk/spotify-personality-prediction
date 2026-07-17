import torch
import torch.nn.functional as F

# CORAL loss function for ordinal tasks
# CORAL converts a K-class ordinal problem into K-1 binary threshold predictions
def coral_loss(logits, y, num_classes=3, class_weights=None):
    y = y.item() if y.dim() == 0 else y.squeeze().item()
    y = torch.tensor(y, device=logits.device, dtype=torch.long)

    loss = 0.0

    for i in range(num_classes - 1):
        # Create binary target for the current threshold
        target = torch.tensor(float(y > i), device=logits.device)

        if class_weights is not None:
            weight = class_weights[int(y.item())]

        else:
            weight = None

        loss += F.binary_cross_entropy_with_logits(logits[i], target, weight=weight)

    return loss / (num_classes - 1)

# Helper function for getting the task losses based on task type
def get_task_loss(task, logits, target, task_info, class_weights=None):
    task_type = task_info[task]["type"] # Define the task type based on task information

    if task_type == "ordinal":
        return coral_loss( # CORAL loss for ordinal types
            logits,
            target,
            num_classes=task_info[task]["n_classes"],
            class_weights=class_weights
        )

    elif task_type == "multiclass":

        return F.cross_entropy( # Cross entropy loss for multi-class types
            logits.unsqueeze(0),
            target.long(),
            weight=class_weights
        )

    elif task_type == "binary":
        pos_weight = None

        if class_weights is not None:
            pos_weight = class_weights[1]

        return F.binary_cross_entropy_with_logits( # BCE with logits for binary types
            logits.squeeze(-1),
            target.float().squeeze(-1),
            pos_weight=pos_weight
        )

# Convert model outputs into predicted class labels
def get_prediction(task, logits, task_info):
    task_type = task_info[task]["type"]

    if task_type == "ordinal":
        probs = torch.sigmoid(logits)

        # Number of thresholds passed determines ordinal class
        pred = torch.sum(probs > 0.5).item()
        
        return pred

    elif task_type == "binary":

        return int(torch.sigmoid(logits).item() > 0.5)

    elif task_type == "multiclass":

        return torch.argmax(logits).item()