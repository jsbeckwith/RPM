# RPM
This repository is the official implementation of https://arxiv.org/abs/2002.06838.
# Balanced-RAVEN
Balanced-RAVEN dataset follows the same setting as the
original RAVEN. Code to generate the dataset resides in the
Datasets/Balanced-RAVEN folder. To generate
a dataset, run python main.py. Check the main.py
file for a full list of arguments you can adjust, e.g.
num-samples is the number of samples per configuration,
save-dir is the directory to save the dataset.
![image](https://github.com/husheng12345/RPM/blob/master/Images/Balanced-RAVEN.png)
# HriNet
Code of our model resides in the HriNet folder.
To train and evaluate our model, run
python main.py. Check the main.py file for a
full list of arguments you can adjust, e.g. dataset is
the dataset to be trained on (PGM or Balanced-RAVEN),
PGM_path and Balanced_RAVEN_path are the path
to the dataset.
![image](https://github.com/husheng12345/RPM/blob/master/Images/HriNet.png)
# Pre-trained models
We provide PyTorch state_dicts (dict of
weight tensors) of our HriNet trained on PGM-70K
and Balanced-RAVEN.
