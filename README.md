# An Innovative 3D Attention Mechanism for Multi-Label Emotion Classification

## Overview

This project is based on the [GoEmotions-pytorch](https://github.com/monologg/GoEmotions-pytorch) repository, implementing an advanced multi-label emotion classification model with a novel 3D Attention Mechanism.

## Taxonomies

The current implementation focuses on the Original GoEmotions dataset, with future plans to expand to:

1. **Original GoEmotions**: 27 emotions + neutral
2. **Hierarchical Grouping**: Positive, negative, ambiguous + neutral
3. **Ekman Emotions**: Anger, disgust, fear, joy, sadness, surprise + neutral

## Requirements

- torch==1.4.0
- transformers==2.11.0
- attrdict==2.0.1
- pandas
- matplotlib
- seaborn
- scikit-learn

## Hyperparameters

### Model Architecture
- Base Model: XLNet (xlnet-base-cased)
- Hidden Size: 768
- Number of Emotions: 28
- Attention Mechanism: Multi-head with emotion-specific attention layers

### Training Configuration
- Optimizer: AdamW
- Learning Rate: 1e-5
- Batch Size: 32
- Epochs: 20
- Loss Function: Binary Cross-Entropy (BCE)
- Device: GPU (CUDA) preferred, fallback to CPU


## Data Preprocessing
- Tokenization: XLNet tokenizer
- Input Sequence Length: 128 tokens
- Train/Test Split: 80/20

## Upcoming Features
- Comprehensive dataset upload
- Data preprocessing scripts
- Ablation study details
