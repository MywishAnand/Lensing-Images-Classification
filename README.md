# Strong Lensing Multi-Class Classification

This project implements a deep learning model using PyTorch to classify gravitational lensing images into three categories:
1. **No Substructure**: Strong lensing images with smoothed mass profiles.
2. **Subhalo Substructure**: Lensing images with subhalo perturbations.
3. **Vortex Substructure**: Lensing images with vortex perturbations.

## Dataset
The dataset consists of normalized gravitational lensing images provided as `.npy` files.

## Project Structure
- `dataset.py`: Custom PyTorch `Dataset` to load `.npy` images.
- `model.py`: ResNet18 model modified for 1-channel grayscale input and 3-class output.
- `train.py`: Script to train the model and save the best weights.
- `evaluate.py`: Script to evaluate the model on the validation set and generate ROC curves.
- `requirements.txt`: Python dependencies.
- `best_model.pth`: Saved weights of the trained model.
- `roc_curve.png`: ROC curves for the three classes.

## Features
- **Transfer Learning**: Uses a pre-trained ResNet18 backbone.
- **Custom Input**: Adapted to handle single-channel `.npy` data.
- **Evaluation**: Multi-class ROC curve analysis with AUC scores.

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Training
Run the training script (ensure the dataset path in `train.py` is correct):
```bash
python train.py
```

### 3. Evaluation
Generate metrics and ROC curves:
```bash
python evaluate.py
```

## Results
The model achieves high AUC scores across all categories:
- **No Substructure**: ~0.975
- **Subhalo Substructure**: ~0.947
- **Vortex Substructure**: ~0.972

![ROC Curve](roc_curve.png)
