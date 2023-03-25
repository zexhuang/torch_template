# pytorch_template

A simple pytorch temple folder for your deep learning project.

## Requirements

- Python >= 3.8.0
- PyTorch >= 1.10
- CUDA >= 11.3 (Optional)

## Folder Structure

```txt
📦pytorch_template
 ┣ 📂cfg
 ┃ ┗ 📜cfg.yaml - configuration for model initialization and training setup
 ┣ 📂data  
 ┃ ┗ 📜dataset.py - dataset loaders
 ┣ 📂exp
 ┃ ┗ 📜eval.ipynb - jupyter notebook for model eval and showing experiment results
 ┣ 📂model
 ┃ ┣ 📜loss.py - custom model loss function 
 ┃ ┗ 📜net.py - base model/neural netowrk class
 ┣ 📂save
 ┃ ┗ ckpt.pth - model checkpoint
 ┣ 📂train
 ┃ ┗ trainer.py - a simple trainer script following torch lightning trainer module
 ┣ 📂utils
 ┃ ┣ 📜metric.py - model evaluation metrics
 ┃ ┗ 📜pytorchtools.py - early stopping class for model training
 ┗ 📜trainval.py - main script to start model training & validation
```
