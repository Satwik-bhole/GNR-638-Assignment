# GNR638: Coding Assignment-2

Pre-trained CNN Representation Transfer and Robustness Analysis

Author: Satwik Pradip Bhole (24B2498) Divyaansh Narkhede (24B0981)

## Overview

This repository contains the implementation for systematically analyzing representation transfer, fine-tuning strategies, few-shot data efficiency, and corruption robustness of pre-trained CNN architectures.
The experiments compare three pre-trained backbones:
* ResNet50
* DenseNet121
* EfficientNet-B0

The models are evaluated on the Aerial Images Dataset (AID) under various controlled scenarios, including linear probing, fractional data training, layer-wise probing, and inference-time distribution shifts (Gaussian noise, motion blur, brightness).

## Repository Structure

    .
    ├── GNR638_Assignment2.ipynb    # Main Jupyter Notebook containing all experiments and outputs
    ├── requirements.txt            # Python dependencies
    ├── README.md                   # Setup and execution instructions
    ├── report.pdf                  # Final 8-12 page technical report
    └── train_data/                       # Directory for the AID dataset (ignored in version control)

## Setup and Execution Instructions

### 1. Create a Virtual Environment
It is highly recommended to use a virtual environment with Python 3.12:

    python3.12 -m venv venv

    # On Linux/macOS:
    source venv/bin/activate

    # On Windows:
    venv\Scripts\activate

### 2. Install Dependencies
    pip install -r requirements.txt

### 3. Prepare the Dataset
Place the downloaded Aerial Images Dataset (AID) into a `data/` directory in the root of the project so it looks like this:

    train_data/
    ├── Airport/
    ├── BareLand/
    ...

### 4. Run the Code

**Method 1: Interactive Execution (Jupyter Notebook)**
Launch the notebook server from your terminal:

    run using run all command of notebook
    or use 
    jupyter notebook

Open `GNR638_Assignment2.ipynb` in your browser, check that the dataset path variable points to your `train_data/` folder, and click **Kernel > Restart & Run All**.

**Method 2: Command Line Execution**
Run the notebook headlessly and save the outputs directly into the file:

    jupyter nbconvert --to notebook --execute GNR638_Assignment2.ipynb
