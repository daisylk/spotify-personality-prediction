# Dictionary for all task information: type, number of classes, class categories
task_info = {
    "openness": {
        "type": "ordinal",
        "n_classes": 3,
        "classes": ["Low", "Medium", "High"]
    },
    "conscientiousness": {
        "type": "ordinal",
        "n_classes": 3,
        "classes": ["Low", "Medium", "High"]
    },
    "extraversion": {
        "type": "ordinal",
        "n_classes": 3,
        "classes": ["Low", "Medium", "High"]
    },
    "agreeableness": {
        "type": "ordinal",
        "n_classes": 3,
        "classes": ["Low", "Medium", "High"]
    },
    "neuroticism": {
        "type": "ordinal",
        "n_classes": 3,
        "classes": ["Low", "Medium", "High"]
    },
    "gender": {
        "type": "multiclass",
        "n_classes": 3,
        "classes": ["Male", "Female", "Other"]
    },
    "age": {
        "type": "ordinal",
        "n_classes": 4,
        "classes": ["13-17", "18-24", "25-30", "31+"]
    },
    "country": {
        "type": "multiclass",
        "n_classes": 10,
        "classes": [
            "Other",
            "United States",
            "Italy",
            "United Kingdom",
            "Canada",
            "Germany",
            "Philippines",
            "Australia",
            "Brazil",
            "India"
        ]
    },
    "marital_status": {
        "type": "binary",
        "n_classes": 2,
        "classes": ["Single", "Relationship"]
    },
    "live_with": {
        "type": "binary",
        "n_classes": 2,
        "classes": ["I live alone", "I live with others"]
    },
    "occupation": {
        "type": "binary",
        "n_classes": 2,
        "classes": ["No job", "Job"]
    },
    "economic": {
        "type": "ordinal",
        "n_classes": 3,
        "classes": ["Low", "Medium", "High"]
    },
    "sport": {
        "type": "ordinal",
        "n_classes": 3,
        "classes": ["No", "Sometimes", "Regularly"]
    },
    "smoke": {
        "type": "binary",
        "n_classes": 2,
        "classes": ["No", "Yes"]
    },
    "alcohol": {
        "type": "binary",
        "n_classes": 2,
        "classes": ["No", "Yes"]
    },
    "spotify_premium": {
        "type": "binary",
        "n_classes": 2,
        "classes": ["No", "Yes"]
    }
}

tasks = list(task_info.keys())