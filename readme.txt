This folder contains source code and a sample dataset for the paper whose title is "Exploiting Representation Curvature for Boundary Detection in Time Series."

To run the source codes, please follow the instructions below.

1. We require following packages to run the code. Please download all requirements in your python environment. 
	- python 3.9.15
	- pytorch 1.13.1
	- numpy 1.25.0
	- pandas 1.4.4
	- cuda 11.7.1 
	- scipy 1.11.1
	- scikit-learn 1.2.2

2. Datasets are in /dataset and should be preprocessed first using preprocessing.ipynb. After preprocessing, datasets are converted into .npy format in /dataset. Now, preprocessed HAPT dataset exists. For 50salads dataset, please download the dataset in this url(https://zenodo.org/record/3625992#.YVwLbdpBx1N).

3. At current directory which has all source codes, run main.py to get AUC and LOC score of RECURVE.
	- dataset: {mHealth, HAPT, WISDM, 50salads}   # designate which dataset to use.
	- seed: {0, 1, 2, 3, 4}	# seed for 5-fold cross validation.
	- gpu: an integer for gpu id
	- repr: {TSCP2, TNC} # representing TPC and TNC representation learning methods
e.g.) python3 main.py --data HAPT --repr TSCP2 --gpu 0 --seed 0

