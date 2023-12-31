# Conversational AI Main Project

This is a code storage for the 2023-2 ConvAI project of 정영준, 송형우, 이윤재, 최서용. 

## Content

1. [Installation](#installation)
2. [Overview](#Overview)
3. [Dataset](#Dataset)
4. [Commands](#Commands)
   
## Installation
Following this [instruction](https://docs.anaconda.com/free/anaconda/install/index.html), please install anaconda virtualenv on your system and then follow the commands below.
```
conda create -n convai python=3.9
conda activate convai
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Overview
This code repo contains a reproduced version of dialog inpainter based on T5 checkpoints. For more information, please check the original paper ([https://arxiv.org/abs/2205.09073](https://arxiv.org/abs/2205.09073)).

## Dataset
There are four datasets in the original paper. Unfortunately, the PublicDialog dataset is not an open source, so we used the rest of datasets, which are [TaskMaster-1](https://huggingface.co/datasets/taskmaster1), [OR-QuAC](https://github.com/prdwb/orconvqa-release), and [QReCC](https://huggingface.co/datasets/voidful/qrecc). Currently, only TaskMaster-1 is used for training the model (will be updated soon).

## Commands
Training
```sh
python run_model.py --data t --batch_size 16 --lr 5e-5 --weight_decay 0.01 --model_name t5-small --epochs 100
```
Inference (Dialog Inpainting) w/o finetuning
```sh
python inference.py --data t --model_name t5-small
```
Inference (Dialog Inpainting) w/ finetuning
```sh
python inference.py --data t --checkpoint_name {checkpoint_name (e.g., epoch-18.pt} --model_name t5-small
```
