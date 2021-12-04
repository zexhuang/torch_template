# pytorch_template
A pytorch temple folder for your deep learning project.

## Requirements
- Python >= 3.8.0
- PyTorch >= 1.10

## Folder Structure
```
📦pytorch_template
 ┣ 📂config
 ┃ ┗ 📜config.yaml - configuration for model initialization and training setup
 ┣ 📂dataloader - load dataset in batch 
 ┃ ┣ 📜image_loader.py 
 ┃ ┗ 📜pointclouds_loader.py
 ┣ 📂experiment
 ┃ ┗ 📜eval.ipynb - jupyter notebook for model eval and showing experiment results
 ┣ 📂model
 ┃ ┣ 📜loss.py - custom model loss function 
 ┃ ┗ 📜net.py - base model/neural netowrk class
 ┣ 📂save
 ┃ ┗ 📜checkpoint.pth - saved model checkpoint
 ┣ 📂utils
 ┃ ┣ 📜metric.py - model evaluation metrics
 ┃ ┗ 📜pytorchtools.py - early stopping class for model training
 ┗ 📜trainval.py - main script to start model training & validation
```
