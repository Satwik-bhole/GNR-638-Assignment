
# GNR638: Coding Assignment-2
## Pre-trained CNN Representation Transfer and Robustness Analysis

**Author:** Satwik Pradip Bhole (24B2498) Divyaansh Narkhede (24B0981)

## Overview
[cite_start]This repository contains the implementation for systematically analyzing representation transfer, fine-tuning strategies, few-shot data efficiency, and corruption robustness of pre-trained CNN architectures[cite: 7]. 

[cite_start]The experiments compare three pre-trained backbones[cite: 24]:
1. [cite_start]**ResNet50** [cite: 25]
2. [cite_start]**DenseNet121** [cite: 28]
3. [cite_start]**EfficientNet-B0** [cite: 29]

[cite_start]The models are evaluated on the **Aerial Images Dataset (AID)**  [cite_start]under various controlled scenarios, including linear probing, fractional data training, layer-wise probing, and inference-time distribution shifts (Gaussian noise, motion blur, brightness)[cite: 41, 68, 82, 97].

## Repository Structure
```text
.
├── GNR638_Assignment2.ipynb    # Main Jupyter Notebook containing all experiments and outputs
├── requirements.txt            # Python dependencies
├── README.md                   # Setup and execution instructions
├── report.pdf                  # Final 8-12 page technical report
└── data/                       # Directory for the AID dataset (ignored in version control)
