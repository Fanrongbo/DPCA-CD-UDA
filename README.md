# code for: DPCA-CD-UDA: Dynamic Multi-Prototype Cross-Attention for Change Detection Unsupervised Domain Adaptation

## Overview

**DPCA-CD-UDA** is an unsupervised domain adaptation (UDA) approach for remote sensing change detection (CD). The model uses a dynamic multi-prototype cross-attention mechanism to bridge the feature distribution gap between source and target domains, improving the performance of change detection models on unlabeled target domains.

The main modules include:

- **Multi-Prototype & Difference Feature Cross-Attention (DMCA)**: This module integrates multi-prototype features to enhance land cover feature representation.
- **Instance Sample Fusion and Pasting (SFP)**: It generates new target domain-style samples of changed regions, assisting in domain adaptation.
- **UDA Optimization Strategy**: The model minimizes the distance between source and target domain features to align them for better change detection performance.

This repository contains the source code for training and evaluating the DPCA-CD-UDA model.

### Hardware Requirements:
- **GPU**: NVIDIA 3090 (24GB VRAM)

### Training the Model

The code can be run via the provided training script. To train the model, use the following command:

```bash
python DAtrain-main.py 
``` 
- **source_path**: Path to the source domain dataset.
- **target_path**: Path to the target domain dataset (unlabeled).
- **epochs**: Total number of epochs for training.

### Model Structure
The DAtrain-main.py script performs the following key steps:

- **Data Loading**: Loads labeled source domain data and unlabeled target domain data.
- **Feature Extraction**: Uses a feature extractor (e.g., FC-Siamese architecture) to extract features from the images.
- **Cross-Attention** Mechanism: Implements the Dynamic Multi-Prototype Cross-Attention module to integrate multi-prototype features.
- **UDA Optimization**: Aligns the source and target domain features using a domain adaptation strategy.
- **Pseudo-Sample Generation**: Generates pseudo-labeled target samples using the Instance Sample Fusion and Pasting (SFP) module.
# Project Overview

This repository contains the implementation of the **DPCA-CD-UDA** model for **Unsupervised Domain Adaptation** in **Change Detection** on remote sensing images. The model leverages dynamic multi-prototype cross-attention mechanisms to improve domain alignment and enhance change detection performance on unlabeled target domains.

The repository includes various Python scripts, each performing a specific task in the training and evaluation pipeline. Below is a detailed description of each file.

## File Descriptions

### 1. `DAtrain-main.py`

This is the main script for training the **DPCA-CD-UDA** model. It handles the following tasks:
- **Data Loading**: Loads labeled source domain data and unlabeled target domain data.
- **Feature Extraction**: Utilizes a feature extractor (such as the FC-Siamese architecture) to extract relevant features from images.
- **Cross-Attention Mechanism**: Implements the **Dynamic Multi-Prototype Cross-Attention** module to enhance feature representation and domain alignment.
- **UDA Optimization**: Aligns source and target domain features using a domain adaptation strategy.
- **Model Training**: Trains the model using both source and target domain data, generating pseudo-labels for target domain images to aid training.
  
### 2. `Appendix-0-FCSiameseOnlySource.py`

This script is responsible for training a **Siamese network** specifically for source domain change detection. It operates without domain adaptation, focusing purely on the source domain for learning the similarity between pairs of images. The Siamese network architecture is typically used for tasks like similarity learning, and in this case, it is applied to change detection in the source domain.

### 3. `Appendix-2-CenterTSNE.py`

This script likely implements **t-SNE** (t-distributed Stochastic Neighbor Embedding) for visualizing the feature space. Specifically, it might apply **Centering** techniques to improve the visualization of high-dimensional features in a 2D or 3D space. This could be used to understand the distributions of features extracted from the source and target domains, as well as the clustering of prototypes.

### 4. `Appendix-3-MultiCenterPeusdoLabel.py`

This script deals with generating **pseudo-labels** for the target domain using a multi-center approach. It uses clustering techniques to create multiple centers in the feature space, which are then used for generating pseudo-labels for the target domain data. These pseudo-labels are essential for unsupervised domain adaptation, where the target domain does not have labeled data.

### 5. `Appendix-4-resnetBGTrain.py`

This file likely contains the training procedure for a **ResNet-based model** for background subtraction in change detection tasks. The **ResNet** architecture is often used for image classification and feature extraction, and in this case, it may be applied to detect changes between different temporal images in remote sensing data.

### 6. `Figure-12-DrawTargetResult.py`

This script is used for **visualizing the target domain results**. It likely takes the output of the change detection model and draws the change detection results on target domain images. The visualizations may include change detection maps or the highlighting of changed regions in images.

### 7. `Figure-3-RetinexDraw.py`

This script seems to implement the **Retinex algorithm**, a technique used for image enhancement and normalization. The script likely focuses on drawing and visualizing the results of applying Retinex-based methods to improve the quality of remote sensing images, particularly for better change detection performance.

### 8. `Figure-7-FCSiameseSourceDrawEntropy.py`

This file seems to be focused on visualizing the results of the **FC-Siamese network** model, specifically on the source domain. The script likely involves drawing results related to entropy measures, possibly to assess uncertainty or variability in the change detection outputs.

## `requirements.txt`

The following is a list of required Python dependencies for the project. You can install them using:

```bash
pip install -r requirements.txt
```

