# Blood Cell Type Prediction Using Deep Neural Networks
🔬 Automated blood cell type prediction using Deep Learning architectures for medical image analysis.

## Overview
This research evaluates six different deep learning architectures for automated classification of blood cell types using the BloodMNIST dataset. The model architectures progress from simple Convolutional Neural Networks (CNNs) to more sophisticated designs including skip-connection networks, CNNs with attention mechanisms (SE/CBAM), Inception-inspired architecture, and Vision Transformer. All models were trained from scratch. 

## Dataset
- **Source**: [BloodMNIST Dataset](https://medmnist.com/)
- **Task**: 8-class classification
- **Format**: Pre-split train/validation/test sets
- **Image size**: 64X64 (RGB)
- **Two distinct Datasets created:** "*Original*" - keeping the imbalance of the classes, "*Augmented*" - data augmentation applied only to train split to address the class imbalance.

## Models
- **Baseline CNN** - shallow CNN model with 2 Conv Layers, each followed by ReLU activation and MaxPooling layer.
- **Baseline CNN with Batch Normalization** - shallow CNN model with 2 Conv Layers, each followed by Batch Normalization, ReLU activation and MaxPooling layer.
- **CNN with Skip Connections:** - ResNet-inspired model with 3 Skip Blocks (each consisting of 2 Residual Units)
- **CNN with Attention Blocks** - Squeeze-and-Excitation (SE) and Convolutional Block Attention Module (CBAM) were separately incorporated into CNN model to evaluate their impact on the prediction capabilities of model.
- **Inception CNN** - inspired by GoogleNet, model architecture incorporates custom Inception-like blocks to enhance representational power by enabling the network to capture features at multiple scales and depths.
- **Visual Transformer** - transformer-based architecture that divides images into patches, applies positional embeddings, and uses multi-head self-attention mechanisms for feature extraction and classification.

## 📊 Results

- **Best Accuracy:** 98.25% (Inception-inspired architecture)
- **Best Efficiency:** 97.84% (CNN + CBAM attention) with lower computational cost
- **Data Augmentation Impact:** 0.46-4.48% improvement across all models

