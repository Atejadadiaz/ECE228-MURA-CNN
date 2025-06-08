# Source Code (`src/`)

This folder contains the core code used to build, train, and evaluate the deep learning models used in the project.

## Files

- `model.py`: Contains the function `get_model()` that loads and modifies CNN architectures such as ResNet18, ResNet50, DenseNet121, and ConvNeXt Tiny, adapting them for binary classification.

- `data.py`: Defines the custom dataset class `MuraDataset`, functions to load and preprocess the MURA dataset (`process_mura_data`), apply image transformations (`get_transforms`), and create dataloaders (`get_dataloaders`).

- `train.py`: Main training script. Defines the `trainer()` function and performs training with validation. Computes loss, accuracy, and Cohen’s kappa. Saves the trained model in the `results/` folder.

## Usage

These files are meant to be imported and used from the notebook or executed directly as standalone scripts. For example:

```python
from src.model import get_model
from src.data import get_dataloaders
from src.train import trainer
