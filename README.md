# Exploiting Representation Curvature for Boundary Detection in Time Series

> __Publication__ </br>
> Shin, Y., Park, J., Yoon, S., Song, H., Lee, B., Lee, J., "Exploiting Representation Curvature for Boundary Detection in Time Series", In *Proceedings of Conference on Neural Information Processing Systems (NeurIPS)*, 2024. [[link]](https://nips.cc/virtual/2024/poster/94837)

This repository is the official PyTorch implementation of **RECURVE**.

<img src='figure.png'>

## How to Install

1. We require following packages to run the code. Please download all requirements in your python environment. 
	- python 3.9.15
	- pytorch 1.13.1
	- numpy 1.25.0
	- pandas 1.4.4
	- cuda 11.7.1 
	- scipy 1.11.1
	- scikit-learn 1.2.2

## Dataset

Datasets are in `/dataset` and should be preprocessed first using `preprocessing.ipynb`. After preprocessing, datasets are converted into .npy format in /dataset. `.npy` files of {HAPT, mHealth, WISDM} is available in the repository. For 50salads dataset, please download the dataset in this [url](https://zenodo.org/record/3625992#.YVwLbdpBx1N).

## How to Run

At current directory which has all source codes, run main.py to get AUC and LOC score of RECURVE.
- dataset: {mHealth, HAPT, WISDM, 50salads}   # designate which dataset to use.
- seed: {0, 1, 2, 3, 4}	# seed for 5-fold cross validation.
- gpu: an integer for gpu id
- repr: {TSCP2, TNC} # representing TPC and TNC representation learning methods
e.g.) python3 main.py --data HAPT --repr TSCP2 --gpu 0 --seed 0

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
