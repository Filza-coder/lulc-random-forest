# 🛰️ LULC Classification with Random Forest — Sentinel-2

> **Multi-class Land Use Land Cover mapping using machine learning on multispectral imagery**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-RandomForest-F7931E?style=flat)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

This project implements a complete **supervised LULC classification pipeline** using Random Forest on Sentinel-2 multispectral data. It covers data preparation, spectral index calculation, model training, accuracy assessment, and map visualization — the full remote sensing ML workflow.

**Classes mapped:** Water | Urban/Built-up | Vegetation | Bare Soil | Agriculture | Forest

---

## 🔬 Methodology

1. **Feature engineering** — 10 Sentinel-2 bands + NDVI, NDWI, NDBI, EVI
2. **Stratified sampling** — 500 pixels per class, train/test split 75/25
3. **Random Forest** — 200 trees, OOB validation
4. **Accuracy assessment** — Overall Accuracy, per-class Precision/Recall/F1, Kappa
5. **Visualization** — Spectral signatures, confusion matrix, feature importance, classified map

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Overall Accuracy | ~93% |
| OOB Score | ~0.92 |
| 5-Fold CV | ~0.92 ± 0.01 |

---

## 🚀 Run

```bash
pip install -r requirements.txt
python lulc_classification.py
```

---

## 📦 Requirements

```
numpy pandas scikit-learn matplotlib scipy
```

---

## 👩‍🔬 Author

**Fatima Filza Hassan** — MS GIS & Remote Sensing, NUST | [Portfolio](https://Filza-coder.github.io)
