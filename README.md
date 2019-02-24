# Galaxy Zoo - The Galaxy Challenge

This repository provides a solution for the [Galaxy Zoo competition in Kaggle](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge). My solution uses the [PyTorch](https://pytorch.org/) open source library for deep learning.

## Instructions to run

### 1. Download the dataset

Download the provided data from the competitions data page:

[https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data).

Unzip the compressed file:
```
unzip galaxy-zoo-the-galaxy-challenge.zip
```

Unzip training and test dataset files:
```
unzip images_test_rev1.zip
unzip images_training_rev1.zip
unzip training_solutions_rev1.zip
unzip all_ones_benchmark.zip
```

### 2. Install the required software

My solution is based on python3 and the pytorch library. To install all the necessary software execute the following:
```
sudo apt-get update && apt-get install python3 python3-pip
sudo pip3 install -r requirements.txt
```

### 3. Download the code from this repository

Clone the current repository:
```
git clone https://github.com/jimsiak/kaggle-galaxies-pytorch.git
cd kaggle-galaxies-pytorch
```

### 4. Run the code

The first thing to do is to modify the configuration appropriately by modifying `config.py`. Then run the code:
```
python3 main.py
```
