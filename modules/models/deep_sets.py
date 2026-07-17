import torch
import torch.nn as nn

# Deep Sets model with multitask prediction heads for all 16 target variables
class DeepSetsModel(nn.Module):
    # Deep Sets architecture with a shared playlist encoder and task-specific output heads
    # The output dimensions are dynamically created based on the task information
    def __init__(self, task_info, input_dim=111, hidden_dim=32): 
        super().__init__()

        # Define phi, the shared playlist encoder/ embedding function
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Task-specific heads
        self.heads = nn.ModuleDict()

        for task, info in task_info.items():

            if info["type"] == "ordinal":
                output_dim = info["n_classes"] - 1 # CORAL uses K-1 logits for K ordinal classes

            elif info["type"] == "binary":
                output_dim = 1 # Single logit for binary classification

            elif info["type"] == "multiclass":
                output_dim = info["n_classes"] # One output logit per class

            self.heads[task] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, output_dim)
            )

    # Forward pass
    def forward(self, x):

        # Encode each playlist
        x = self.phi(x) # (1, n_playlists, 111) > (1, n_playlists, 32)

        # Mean pooling - averages over playlists and retains permutation invariance
        x = torch.mean(x, dim=1) # (1, n_playlists, 32) > (1, 32)

        # Pass shared user representation through each personality head - one prediction head per task
        outputs = {}

        for task, head in self.heads.items():
            outputs[task] = head(x)
        
        return outputs