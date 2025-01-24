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

We follow the previous work [DiFashion](https://github.com/YiyanXu/DiFashion?tab=readme-ov-file) and use the datasets of iFashion and Polyvore-U, which include the required data of both fashion outfit and user-fashion item interactions. 

## Fashion Image Generation without Feedback

During the sampling phase, this code generates 7 recommended items for each outfit as a candidate set. 
```
cd ./fashiondpo
python sample.py
```
If it is the first round of sampling in the iterative process, set `args.resume=False`, which will create the LoRA layers in the pre-trained model. In subsequent rounds, set `args.resume=True` to load the fine-tuned LoRA layers from the model specified in args.pretrained_model_name_or_path.

## Feedback Generation from Multiple Experts

### Get Feedback
We locally deploy MiniCPM to evaluate "Quality". For "Compatibility", we train a VBPR model using paired outfit data from the POG and Polyvore-U dataset. "Personalization" is evaluated using the CLIP Score.
```
cd ./evaluation
python multiple_evaluate.py
```
This contains three evaluating perspectives:

1) Quality: Loading the per-trained [MiniCPM-Llama3-V 2.5](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5).
The prompt:
```
Consider whether the fashion elements in the image are complete and whether they conform to fashion design principles. The goal is to classify the quality into one of the following categories: 1-Very Poor Quality, 2-Poor Quality, 3-Low Quality, 4-Below Average Quality, 5-Moderate Quality, 6-Above Average Quality, 7-Good Quality, 8-Very Good Quality, 9-High Quality, 10-Exceptional Quality. Please provide the best possible category based on the available information. 
```
Set `aegs.evanumber=5 or 10` to specify the number of classification levels to be used in the MiniCPM evaluation.


2) Compatibility: We trained the VBPR model using the iFashion dataset. The checkpoint is avaliable at [vbpr_compatibility](https://huggingface.co/AZhe1220/fashiondpo/tree/main).
If you want to retrain the model, please refer to `train_VBPR.py.`

4) Personalization: We use the pre-trained [CLIP (ViT-B/32)](https://huggingface.co/sentence-transformers/clip-ViT-B-32) to encode the generated fashion items and user interaction history image items, and calculate the CLIP Score between them.


Normalize the scores and make preference judgments to construct positive-negative pairs:
```
cd ./evaluation
python analyse.py
```

### Model Fine-tuning with Direct Preference Optimization

In each outfit, if any two items from the 7 candidate generated items form a preference-non-preference relationship, we perform LoRA fine-tuning on the 50 saved timesteps:
```
cd ./fashiondpo
python finetune.py
```

### Release checkpoint

The checkpoints are released via [the anonymous huggingface repo](https://huggingface.co/AZhe1220/fashiondpo/tree/main). `checkpoint_ifashion` is fine-tuned on the iFashion dataset, while `checkpoint_polyvore` is fine-tuned on the Polyvore dataset.


