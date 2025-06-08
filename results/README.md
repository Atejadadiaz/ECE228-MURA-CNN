# Results (`results/`)

This folder contains outputs generated during training and evaluation of the models.

## Contents

- `*.pt`: Trained model weights (ResNet, DenseNet, ConvNeXt)
- `*.png`: Plots comparing model performance:
  - `train_accuracy_comparison.png`
  - `val_accuracy_comparison.png`
  - `kappa_score_comparison.png`

## Model Weights (Google Drive)

Due to GitHub file size limits, the trained model weights are available via Google Drive:

- [resnet18_MURA.pt](https://drive.google.com/file/d/1y0qdmourmeDp-jZTbXXZt-awjMNsVKKq/view?usp=drive_link)  
- [resnet50_MURA.pt](https://drive.google.com/file/d/1OBO2XzwUAWSKHp6jxc2CkK9pXtQuxmTV/view?usp=drive_link)  
- [densenet121_MURA.pt](https://drive.google.com/file/d/1NPt7E0UI_3Tt9tnkBiNNLvMapWKBzUSd/view?usp=drive_link)  
- [convnext_MURA.pt](https://drive.google.com/file/d/1UWa8jiTwtNfJfX7gVM1Qh3axmWSaz3PI/view?usp=drive_link)

> Download the desired model and place it inside this `results/` folder to run evaluation or make predictions.

## How to Load a Saved Model

```python
from src.model import get_model
import torch

model = get_model("resnet18")
model.load_state_dict(torch.load("results/resnet18_MURA.pt"))
model.eval()
