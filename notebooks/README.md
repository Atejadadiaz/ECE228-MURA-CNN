# Jupyter Notebooks (`notebooks/`)

This folder contains the main notebook used to run and evaluate the models in this project.

## File

- `ECE228_Project.ipynb`: Main Colab notebook that handles the full pipeline:
  - Load the dataset (from Google Drive)
  - Apply data transformations
  - Load CNN model (ResNet, DenseNet, etc.)
  - Train and validate the model
  - Evaluate results and plot training curves
- `compare_models.ipynb`: Visualizes and compares training and validation accuracy and Kappa score across different model architectures using the saved results.

## How to Use in Google Colab

1. Open `ECE228_Project.ipynb` in [Google Colab](https://colab.research.google.com).
2. Mount your Google Drive in the first code cell:

```python
from google.colab import drive
drive.mount('/content/drive')
