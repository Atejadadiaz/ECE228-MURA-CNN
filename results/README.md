# Results (`results/`)

This folder contains the outputs generated during training and evaluation of the models.

## Contents

- `*.pth`: Saved model weights after training. For example:
  - `resnet18_mura.pth`
  - `densenet121_mura.pth`

- (Optional) `*.png`: Visualizations such as loss/accuracy curves or ROC curves.

- (Optional) `metrics.csv` or `metrics.json`: Evaluation metrics such as accuracy, precision, recall, AUC, and Cohen’s kappa.

## How to Use

To save a trained model:

```python
torch.save(model.state_dict(), 'results/resnet18_mura.pth')
```

To load it again later:
```bash
model.load_state_dict(torch.load('results/resnet18_mura.pth'))
```
