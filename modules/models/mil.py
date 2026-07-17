import torch
import torch.nn as nn

# Multiple Instance Learning (MIL) model with multitask prediction heads for all 16 target variables
class MILModel(nn.Module):
    # MIL architecture with a shared playlist encoder and task-specific output heads, with attention pooling
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
            nn.Dropout(0.2)
        )

        # Define the attention layer - computes an importance score for each playlist
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(), # Hyperbolic tangent activation function, bounds scores between -1, 1, prevents extremes
            nn.Linear(32, 1)
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
    def forward(self, x, return_attention=False):

        # Encode each playlist
        x = self.phi(x) # (1, n_playlists, 111) > (1, n_playlists, 32)

        # Compute raw attention scores and normalise (so they sum to 1) via softmax
        weights = self.attention(x).squeeze(-1)
        weights = torch.softmax(weights, dim=1)

        # Weighted aggregation of instance embeddings - one user representation
        pooled = torch.sum(weights.unsqueeze(-1) * x, dim=1)

        # Pass shared user representation through each personality head - one prediction head per task
        outputs = {}

        for task, head in self.heads.items():
            outputs[task] = head(pooled)

        if return_attention:
            return outputs, weights
        
        return outputs