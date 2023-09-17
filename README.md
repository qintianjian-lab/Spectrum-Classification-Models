# Spectrum-Classification-Models

![GitHub](https://img.shields.io/github/license/qintianjian-lab/Spectrum-Classification-Models?style=flat-square)

This repository contains the implementation of several spectrum classification models.

## Implementation Models

1. SSCNN `doi: 10.1093/mnras/sty3020`
2. RAC-Net (Including C-Net & RC-Net) `doi: 10.1088/1538-3873/ab7548`
3. [ConvNeXt](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html) for
   1D

## Installation

```bash
# Get the code from repository
git clone https://github.com/qintianjian-lab/Spectrum-Classification-Models.git
cd Spectrum-Classification-Models

# Install dependencies
# 1. Install PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# or follow the official guide: https://pytorch.org/get-started/locally/

# 2. Install other dependencies
pip3 install -r requirements.txt
```

## Usage

Read `config/config.py` for more details.

## Dataset Directory Structure

Support K-fold cross-validation.

```
├── DATASET
│   ├── fold 1
│   │   ├── train
│   │   │   ├── xxx 1.csv
│   │   │   ├── xxx 2.csv
│   │   │   └── ...
│   │   ├── val
│   │   │   ├── yyy 1.csv
│   │   │   ├── yyy 2.csv
│   │   │   └── ...
│   │   ├── test
│   │   │   ├── zzz 1.csv
│   │   │   ├── zzz 2.csv
│   │   │   └── ...
│   ├── fold 2
│   │   ├── ...
│   ├── fold 3
│   │   ├── ...
└── ...
```

