import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Preprocessing steps
# Loading playlist and user data
def load_data():
    url_playlists = "https://raw.githubusercontent.com/pierz95/SpotifyAttributes/refs/heads/main/Datasets/df_playlists.csv"
    url_users = "https://raw.githubusercontent.com/pierz95/SpotifyAttributes/refs/heads/main/Datasets/user_targets.csv"

    playlists = pd.read_csv(url_playlists)
    users = pd.read_csv(url_users)

    return playlists, users

# Drop identifier columns
def get_playlist_feature_cols(playlists):
    playlist_feature_cols = playlists.columns.drop(
        ["Unnamed: 0", "id_playlist", "id_owner"]
    )

    return playlist_feature_cols

# Clean whitespaces and correct column names
def clean_users(users):
    users = users.rename(
        columns={"alchol": "alcohol"}
    )

    users.columns = users.columns.str.strip()

    return users

# Encode categorical labels as integers
def encode_users(users):
    binary_mapping = {
        "No": 0,
        "Yes": 1
    }

    marital_mappings = {
        "Single": 0,
        "Relationship": 1
    }

    livewith_mappings = {
        "I live alone": 0,
        "I live with others": 1
    }

    occupation_mappings = {
        "No job": 0,
        "Job": 1
    }

    gender_mapping = {
        "Male": 0,
        "Female": 1,
        "Other": 2
    }

    country_mapping = {
        "Other": 0,
        "United States": 1,
        "Italy": 2,
        "United Kingdom": 3,
        "Canada": 4,
        "Germany": 5,
        "Philippines": 6,
        "Australia": 7,
        "Brazil": 8,
        "India": 9
    }

    economic_mappings = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }

    sport_mappings = {
        "No": 0,
        "Sometimes": 1,
        "Regularly": 2
    }

    users["smoke"] = users["smoke"].map(binary_mapping)
    users["alcohol"] = users["alcohol"].map(binary_mapping)
    users["spotify_premium"] = users["spotify_premium"].map(binary_mapping)
    users["marital_status"] = users["marital_status"].map(marital_mappings)
    users["live_with"] = users["live_with"].map(livewith_mappings)
    users["occupation"] = users["occupation"].map(occupation_mappings)
    users["gender"] = users["gender"].map(gender_mapping)
    users["country"] = users["country"].map(country_mapping)
    users["economic"] = users["economic"].map(economic_mappings)
    users["sport"] = users["sport"].map(sport_mappings)

    return users 

# Create a lookup table which maps the user IDs to its target labels
def create_target_lookup(users):
    target_lookup = {}

    for _, row in users.iterrows():
        target_lookup[row["id_owner"]] = {
            "openness": row["openness"],
            "conscientiousness": row["conscientiousness"],
            "extraversion": row["extraversion"],
            "agreeableness": row["agreeableness"],
            "neuroticism": row["neuroticism"],
            "gender": row["gender"],
            "age": row["age"],
            "country": row["country"],
            "marital_status": row["marital_status"],
            "live_with": row["live_with"],
            "occupation": row["occupation"],
            "economic": row["economic"],
            "sport": row["sport"],
            "smoke": row["smoke"],
            "alcohol": row["alcohol"],
            "spotify_premium": row["spotify_premium"]
        }

    return target_lookup

# Group playlists by user
def group_playlists(playlists):
    grouped = playlists.groupby("id_owner")

    return grouped

# Create the user-level dataset
def create_dataset(grouped, playlist_feature_cols, target_lookup):
    dataset = []

    for user_id, user_data in grouped:

        # Playlist features for the current user
        X_user = user_data[playlist_feature_cols].values

        # Target labels for the current user
        targets = target_lookup[user_id]

        dataset.append(
            (X_user, targets)
        )

    return dataset

# Run the complete preprocessing pipeline
def prepare_dataset():

    playlists, users = load_data()

    playlist_feature_cols = get_playlist_feature_cols(playlists)

    users = clean_users(users)
    users = encode_users(users)

    target_lookup = create_target_lookup(users)

    grouped = group_playlists(playlists)

    dataset = create_dataset(grouped, playlist_feature_cols, target_lookup)

    return dataset 

# Standardise playlist features using the training fold
def scale_dataset(train_data, val_data, test_data):
    scaler = StandardScaler()

    X_train_all = np.vstack([x for x, y in train_data])

    scaler.fit(X_train_all)

    def scale(data):
        return [(scaler.transform(x), y) for x, y in data]

    train_scaled = scale(train_data)
    val_scaled = scale(val_data)
    test_scaled = scale(test_data)

    return train_scaled, val_scaled, test_scaled