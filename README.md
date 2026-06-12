# A Few-Shot Siamese Network with Adaptive Metric Learning for Printed Circuit Board Defect Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange?logo=pytorch)
![Model](https://img.shields.io/badge/Backbone-YOLOv8%20%7C%20ResNet--50%20%7C%20EfficientDet-green)
![Task](https://img.shields.io/badge/Task-Few--Shot%20Object%20Detection-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **FSODM-Siamese**: A backbone-free Siamese twin network integrating adaptive metric learning for detecting PCB defects under limited labelled data conditions.

---

## 📌 Overview

Automated optical inspection of Printed Circuit Boards (PCBs) faces a critical bottleneck: acquiring large annotated datasets is costly and impractical in real manufacturing environments. This project proposes **FSODM-Siamese**, a few-shot object detection framework that learns to detect six PCB defect categories from as few as **K=1 or K=5 labelled examples per class**.

The framework employs a **metric learning paradigm** in which a Siamese twin network learns discriminative feature embeddings to measure similarity between query and support PCB images. A comparative evaluation of three backbone architectures — **EfficientDet**, **ResNet-50**, and **YOLOv8** — within twin network configurations identifies a hybrid model as the best-performing solution.

---

## 🔍 Defect Categories

The model is evaluated across six PCB defect types from the DeepPCB benchmark dataset:

| Defect Class     | Description                                      |
|------------------|--------------------------------------------------|
| Missing Hole     | Absence of a required drill hole                 |
| Mouse Bite       | Irregular notch on the PCB edge                  |
| Open Circuit     | Broken conductive path                           |
| Short            | Unintended connection between two conductors     |
| Spur             | Unwanted copper protrusion                       |
| Spurious Copper  | Extra copper remaining after etching             |

---

## 🏗️ Architecture

The **FSODM-Siamese** framework consists of the following components:

- **Backbone-Free Siamese Twin Network** — trained from scratch with shared weights across both streams
- **Five Feature Representation Variants** — including raw embeddings, attention-weighted, and hybrid representations
- **Four Attention Mechanisms** — including the novel **Hybrid Spatial-Channel Attention (HSCA)** module
- **Annotation-Aware Learning** — per-instance confidence weighting during training
- **MAML-Style Meta-Learning Strategy** — episodic training with N-way K-shot tasks

### Backbone Configurations Evaluated

| Backbone      | Input Resolution | Role in Siamese Stream |
|---------------|-----------------|------------------------|
| YOLOv8        | 640×640         | Feature extractor       |
| ResNet-50     | 224×224         | Feature extractor       |
| EfficientDet  | 512×512         | Feature extractor       |

---

## 📊 Results

Best performance achieved at **K=5** shots:

| Metric       | Score   |
|--------------|---------|
| Accuracy     | 96.88%  |
| Macro F1     | 0.9661  |

### Comparison of Few-Shot Methods

| Method  | Backbone  | K=1 Acc (%) | K=5 Acc (%) |
|---------|-----------|-------------|-------------|
| FSODM   | YOLOv8    | —           | **96.88**   |
| TFA     | ResNet-50 | —           | —           |
| DSRF    | ResNet-50 | —           | —           |
| VFA     | EfficientDet | —        | —           |

> Full per-class and per-method results are available in [`results.csv`](./results.csv).

---

## 📁 Repository Structure

```
├── pcb_siamese_yolov8 few shot.ipynb   # Main training & evaluation notebook (Google Colab)
├── pcb_siamese_yolov8 few shot.py      # Standalone Python script version
├── results.csv                          # Experiment results across backbones and K-shot settings
└── README.md
```

---

## ⚙️ Setup & Requirements

### Prerequisites

```bash
Python >= 3.8
torch >= 2.0
torchvision
ultralytics       # YOLOv8
numpy
pandas
scikit-learn
matplotlib
opencv-python
```

### Installation

```bash
git clone https://github.com/Denuwanweerakkody/A-Few-Shot-Siamese-Network-with-Adaptive-Metric-Learning-for-Printed-Circuit-Board-Defect-Detection.git
cd A-Few-Shot-Siamese-Network-with-Adaptive-Metric-Learning-for-Printed-Circuit-Board-Defect-Detection
pip install -r requirements.txt
```

### Running on Google Colab

The primary implementation runs on **Google Colab** with GPU support. Open the notebook directly:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Denuwanweerakkody/A-Few-Shot-Siamese-Network-with-Adaptive-Metric-Learning-for-Printed-Circuit-Board-Defect-Detection/blob/main/pcb_siamese_yolov8%20few%20shot.ipynb)

---

## 📦 Dataset

This project uses the **DeepPCB** dataset containing PCB defect images across six categories.

- **Total images**: ~1,384
- **Split strategy**: Stratified 80% train / 10% validation / 10% test
- **Few-shot episodes**: N-way K-shot tasks sampled from training split

> Dataset is not included in this repository. Please download from the [DeepPCB GitHub](https://github.com/tangsanli5201/DeepPCB) and place under `data/`.

---

## 🧠 Key Contributions

1. **FSODM-Siamese Framework** — a novel integration of few-shot meta-learning with Siamese architecture for PCB inspection
2. **Hybrid Spatial-Channel Attention (HSCA)** — a new attention module combining spatial and channel cues for defect localisation
3. **Annotation-Aware Loss** — per-instance confidence weighting to handle label noise in small datasets
4. **Backbone Comparative Study** — systematic evaluation of YOLOv8, ResNet-50, and EfficientDet under identical few-shot protocols

---

## 📝 Citation

If you use this work, please cite:

```bibtex
@article{weerakkody2025fsodm,
  title     = {A Few-Shot Siamese Network with Adaptive Metric Learning for Printed Circuit Board Defect Detection},
  author    = {Weerakkody, Denu and Osagie, Efosa},
  year      = {2025},
  note      = {Preprint}
}
```

---

## 👥 Authors

- **Denu Weerakkody** — [@Denuwanweerakkody](https://github.com/Denuwanweerakkody)
- **Efosa Osagie** — Collaborator

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
