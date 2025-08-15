# Landmark-classification-Tagging-for-Social-Media

## Overview
This project implements two approaches to classify images of landmarks into **50 distinct categories**:

1. **Custom VGG-like CNN** built from scratch for feature extraction and classification.  
2. **Transfer Learning** using a pre-trained convolutional neural network to boost accuracy.

The dataset contains diverse landmark images, where some are visually subtle and challenging to distinguish, making this a non-trivial image classification problem.

---

## Project Structure
├── cnn_from_scratch.ipynb # Main Jupyter notebook with full workflow
├── src/
│ ├── data.py # Data loading, preprocessing, and augmentation
│ ├── model.py # Model architecture definitions
│ ├── train.py # Training and evaluation logic
├── static_images/ # Sample images and visualization assets
├── requirements.txt # Python dependencies


---

## Dataset
- **Source:** Provided through the Udacity workspace  
- **Classes:** 50 landmark categories  
- **Splits:** Train / Validation / Test  
- **Image Size:** Resized and center-cropped to **244×244** pixels  
- **Augmentation:** Random horizontal flips for training data  

---

## Models

### 1. VGG-like CNN (from scratch)
- Multiple convolutional layers with ReLU activation  
- Max-pooling for spatial downsampling  
- Fully connected layers with dropout  
- Adaptive Average Pooling before flattening to handle varying input sizes  

### 2. Transfer Learning
- Base: Pre-trained convolutional network from `torchvision.models`  
- Custom classifier replacing the original final layer  
- Fine-tuning strategy to leverage pre-trained weights for landmark recognition  

---

## Training
- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** Adam  
- **Batch Size:** 64  
- **Hardware:** GPU-enabled training  
- **Stopping Criteria:** Fixed number of epochs, monitored validation accuracy  

---

## Results
| Model                  | Test Accuracy |
|------------------------|--------------:|
| VGG-like (from scratch) | ~50%+         |
| Transfer Learning       | Higher than scratch model (74%) |

---
Key Learnings

Building a deep CNN from scratch requires careful architecture design and data augmentation to avoid overfitting.

Transfer learning significantly improves performance with less training time and smaller datasets.

