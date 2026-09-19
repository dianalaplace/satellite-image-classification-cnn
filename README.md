# Satellite Land-Cover Classification with a Custom CNN (PyTorch)

A convolutional neural network built from scratch in PyTorch that classifies 64×64 RGB satellite image patches into 10 land-cover classes (EuroSAT-style): residential, industrial, agricultural, and natural terrain.

**Validation accuracy: 94.1%** across 10 balanced classes.

## What's in the notebook

- Custom `Dataset` / `DataLoader` pipeline with on-the-fly augmentation (flips, rotation, color jitter)
- A compact CNN (4 conv blocks with BatchNorm, global average pooling, dropout) trained from scratch — no pretrained backbone
- Training loop with early stopping and `ReduceLROnPlateau` scheduling
- Full evaluation: per-class precision/recall/F1, confusion matrix, and misclassified-sample inspection
- EDA: per-channel pixel distributions and per-class brightness comparison

## Classes

`AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, `SeaLake`

## Stack

Python, PyTorch, torchvision, pandas, scikit-learn, matplotlib, seaborn

## Notes

The notebook expects `data/` (training images, one subfolder per class) and `public_test_data/` (unlabeled test images) in the working directory. These aren't included here — swap in any EuroSAT-style dataset with the same folder layout to reproduce.
