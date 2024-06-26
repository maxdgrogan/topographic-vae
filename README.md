# Variational Autoencoders with Topographic Latent Spaces

This repository contains the codebase for training variational autoencoders with topographic latent spaces such as those used in [Predicting Proprioceptive Cortical Anatomy and Neural Coding with Topographic Autoencoders (2022)](https://www.biorxiv.org/content/10.1101/2021.12.10.472161v3).

## Installation

The code for this repository can be downloaded using
```
git clone git@github.com:maxdgrogan/topographic-vae.git
```

or from https://doi.org/10.6084/m9.figshare.c.5762372.v1.

To install the required libraries with `pip`, navigate to the `topographic_vae` and run:
```
pip install -r requirements.txt
```
You will also need to install PyTorch (version 1.9.1) either with cuda support for GPU training:
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```
or without cuda support (if you plan to train on CPU only):
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```

Once you have installed the required libraries, install the repository codebase by using:
```
pip install -e .
```

## Training topographic VAEs

All code is documented and a notebook tutorial is provided in `tutorial.ipynb`. For further details please refer to the *Methods* sections of [Blum et al 2022](https://www.biorxiv.org/content/10.1101/2021.12.10.472161v3).

This notebook allows users to train models on both dummy data and the kinematic data used in [Blum et al 2022](https://www.biorxiv.org/content/10.1101/2021.12.10.472161v3) which can be downloaded [here](https://figshare.com/articles/dataset/Behavioural_kinematics_and_S1_neural_activity_datasets/17700944). The kinematic data directory should have the following structure:

```
└── data
    └── kinematic               <- Data downloaded from figshare.
        ├── natural             <- Natural kinematic data.
        │   ├── C09-001.h5      <- Single-subject natural kinematic data.
        │   └── ...
        └── planar
            └── planar_data.h5  <- Planar reaching kinematic data.
``` 

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).



