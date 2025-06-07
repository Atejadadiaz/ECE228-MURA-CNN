# ECE228-MURA-CNN

This project implements a convolutional neural network (CNN) to detect musculoskeletal abnormalities in X-ray images using the MURA dataset. It is part of the final project for the ECE 228 Machine Learning course at UCSD.

## Objective

Develop a CNN to detect musculoskeletal abnormalities in X-rays using the MURA dataset.

## Repository Structure
- `data/`: Scripts or instructions to obtain and preprocess the MURA dataset.
- `notebooks/`: Jupyter notebooks for data exploration and model evaluation.
- `results/`: Output results, including trained models and evaluation metrics.
- `src/`: Source code for model architectures, training routines, and utilities.
- `README.md`: Project description and instructions

## Installation

1. **Clone this repository**:

   bash
   git clone https://github.com/Atejadadiaz/ECE228-MURA-CNN.git
   cd ECE228-MURA-CNN
   
2. **Install dependencies**:
   bash
   pip install -r requirements.txt

## Dataset

This project uses the [MURA dataset](https://stanfordmlgroup.github.io/competitions/mura/), a large collection of musculoskeletal radiographs.

To run the code:

1. Download the dataset manually from the official Stanford MURA website.
2. Unzip the folder `MURA-v1.1` inside the `data/` directory (or update the paths in the code if you store it elsewhere).

> Note: Due to size and license restrictions, the dataset is not included in this repository.

## Running the Code

There are two ways to run this project:

### 📘 Option 1: Use the Jupyter Notebook

Run the notebook below to train, evaluate, and visualize the results:

notebooks/Project_Code.ipynb

> Make sure the MURA dataset is correctly placed and the required libraries are installed.

### 💻 Option 2: Use scripts 

To train the model (example):

   ```bash
   python src/train.py
   python src/evaluate.py --model_path results/model.pth


> This project currently works entirely from the notebook. Scripts can be added later if needed.

### Results & Reproducibility

* All training and evaluation steps are included in notebooks/Project_Code.ipynb.

* Trained model weights and generated plots are stored in the results/ folder.

* You can reload saved models using:

   ```bash
   import torch
   model.load_state_dict(torch.load('results/model.pth'))
   model.eval()


> Ensure the same model architecture is used when loading saved weights.

This setup allows full reproducibility of the results shown in the final report.
