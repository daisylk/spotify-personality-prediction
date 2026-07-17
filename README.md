# Music Listening Behaviour and Personality: Predicting Psychological, Demographic, and Lifestyle Characteristics from Spotify Data.

## Overview
This project aims to investigate whether behavioural signals extracted from Spotify playlists can be used to predict personality traits, demographic characteristics, and lifestyle factors. Two multitask deep learning approaches are used, Deep Sets and attention-based Multiple Instance Learning. Playlist-level features are aggregated into user-level representations, allowing prediction of 16 target variables. 

## Repository Structure
```text
├── modules/
│   ├── data/
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── deep_sets.py
│   │   └── mil.py
│   ├── training/
│   │   ├── cross_validation.py
│   │   ├── evaluation.py
│   │   ├── losses.py
│   │   └── trainer.py
│   ├── results/
│   ├── run_deepsets.py
│   ├── run_mil.py
│   └── task_info.py
├── .gitignore
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Dataset
The dataset was originally collected by Tricomi et al. (2024) and is accessible at [SpotifyAttributes dataset](https://github.com/pierz95/SpotifyAttributes). There are 10,286 public playlists created by 739 users, with 111 features aggregated from song-level to playlist-level. The playlist-level dataset includes features such as audio characteristics, artist features, genre information, and popularity measures. The user-level dataset includes personality traits, demographic characteristics, and lifestyle factors. 

## Models
Two neural network architectures are implemented, Deep Sets and Multiple Instance Learning. Both methods are designed to handle set-structured datasets with varying numbers of instances per set while maintaining permutation invariance. Deep Sets uses a shared playlist encoder followed by mean pooling to create a user-level representation. Multiple Instance Learning uses attention-based pooling, which assigns weights to each instance based on learned importance to create a user-level representation. Both models use multitask learning to predict all 16 targets simultaneously.

## Installation
This project was developed using Python 3.13.0.
Clone the repository and install dependencies:

```bash
git clone https://github.com/daisylk/spotify-personality-prediction.git
cd spotify-personality-prediction
pip install -r requirements.txt
```

## Usage
The datasets are automatically loaded from the original SpotifyAttributes repository by the preprocessing script. To run the two main scripts:

```bash
python modules/run_deepsets.py
```
```bash
python modules/run_mil.py
```

Results will be stored inside the `modules/results/` folder as `.pkl` files. The file name for results can be changed by editing the second-to-last line in each script prior to running the models.

## Results
Both models were found to perform similarly across the 16 prediction tasks. Some of the variables were comparatively easier to predict, while some tasks performed at no better than chance level. 

## References
- Tricomi, P. P., Pajola, L., Pasa, L., & Conti, M. (2024). “All of Me”: Mining Users’ Attributes from their Public Spotify Playlists. In Companion Proceedings of the ACM Web Conference 2024 (WWW ’24 Companion) (963-966). Association for Computing Machinery. https://doi.org/10.1145/3589335.3651459
- Zaheer, M., Kottur, S., Ravanbakhsh, S., Póczos, B., Salakhutdinov, R., & Smola, A. J. (2017). Deep Sets. *Advances in Neural Information Processing Systems*, 30. https://proceedings.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html
- Ilse, M., Tomczak, J. M., & Welling, M. (2018). Attention-based deep multiple instance learning. In J. Dy & A. Krause (Eds.), *Proceedings of the 35th International Conference on Machine Learning* (Vol. 80, pp. 2127–2136). PMLR. https://proceedings.mlr.press/v80/ilse18a.html