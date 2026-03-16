
# GNR638: Coding Assignment-2
## Pre-trained CNN Representation Transfer and Robustness Analysis

**Author:** Satwik Pradip Bhole (24B2498) Divyaansh Narkhede (24B0981)

## Overview
This repository contains the implementation for systematically analyzing representation transfer, fine-tuning strategies, few-shot data efficiency, and corruption robustness of pre-trained CNN architectures. 

The experiments compare three pre-trained backbones:
1. **ResNet50** 
2. **DenseNet121**
3. **EfficientNet-B0**

The models are evaluated on the **Aerial Images Dataset (AID)** under various controlled scenarios, including linear probing, fractional data training, layer-wise probing, and inference-time distribution shifts (Gaussian noise, motion blur, brightness).

## Repository Structure
```text
.
├── GNR638_Assignment2.ipynb    # Main Jupyter Notebook containing all experiments and outputs
├── requirements.txt            # Python dependencies
├── README.md                   # Setup and execution instructions
├── report.pdf                  # Final 8-12 page technical report
└── data/                       # Directory for the AID dataset (ignored in version control)
