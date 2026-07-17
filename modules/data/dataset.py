import torch
from torch.utils.data import Dataset

# Dataset for user-level Spotify playlist data
# Each sample returned consists of a variable-length playlist matrix and a target label dictionary
class SpotifyDataset(Dataset):

    # Initialisation of dataset
    def __init__(self, data):
        self.data = data

    # Number of users
    def __len__(self):
        return len(self.data)

    # Define how to get one sample
    def __getitem__(self, idx):

        X_user, targets = self.data[idx]

        # Convert NumPy arrays to PyTorch tensors
        X_user = torch.tensor(X_user, dtype=torch.float32)
        targets = {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in targets.items()
        }

        return X_user, targets