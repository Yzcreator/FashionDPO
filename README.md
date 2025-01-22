# FashionDPO

This is the implementation of FashionDPO.

![Results for this project](figure/model_result.png)

## TODO List
- [x] Environment
- [x] Datasets
- [x] Fashion Image Generation without Feedback
- [x] Feedback Generation from Multiple Experts
- [x] Model Fine-tuning with Direct Preference Optimization
- [x] Release checkpoint

## Installation
Clone this repository:
```
git clone https://github.com/Yzcreator/FashionDPO.git
cd ./FashionDPO/
```
Install PyTorch and other dependencies:
```
conda env create -f fashiondpo_environment.yml
conda activate FashionDPO
```

## Datasets

We follow the previous works DiFashion and use the datasets of iFashion~\cite{POG} and Polyvore-U~\cite{polyvore_u}, which include the required data of both fashion outfit and user-fashion item interactions.

