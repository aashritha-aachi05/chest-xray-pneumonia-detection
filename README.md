# Chest X-Ray Pneumonia Detection

**CS439 Final Project**

A hybrid machine learning pipeline for pneumonia detection in chest X-rays, combining unsupervised K-Means segmentation with supervised classification, evaluated for cross-dataset generalization.

---

## Project Overview

This project investigates whether K-Means-based image features learned on one chest X-ray dataset (Kaggle) generalize to a completely different dataset (NIH ChestX-ray14). We compare four classifiers trained on K-Means features against a raw-pixel baseline, with the tuned XGBoost model achieving the best cross-dataset performance.

### Key Contributions
1. **K-Means pre-screening pipeline** — unsupervised segmentation of X-ray images into background, lung tissue, and abnormality regions, followed by compact feature extraction
2. **Multi-model comparison** — Logistic Regression, Random Forest, SVM, and XGBoost (with RandomizedSearchCV tuning)
3. **Cross-dataset generalization study** — models trained on Kaggle evaluated on NIH ChestX-ray14, measuring the real-world performance gap
4. **Ablation study** — K-Means features vs. raw PCA pixel features

---

## Repository Structure

```
chest-xray-pneumonia-detection/
├── 1_preprocessing.ipynb       # Data loading, resizing, normalization
├── 2_kmeans_segmentation.ipynb # K-Means clustering + feature extraction
├── 3_classification.ipynb      # Model training, tuning, evaluation
├── 4_generalization.ipynb      # Cross-dataset generalization on NIH
├── 5_visualizations.ipynb      # All figures used in the report
├── report.pdf                  # Final NeurIPS-formatted report
└── README.md
```

---

## Datasets

| Dataset | Source | Images | Use |
|---------|--------|--------|-----|
| Kaggle Chest X-Ray (Pneumonia) | [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) | 5,856 | Training + in-distribution test |
| NIH ChestX-ray14 (subset) | [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) | ~2,000 | Out-of-distribution generalization test |

---

## How to Reproduce

### 1. Setup
```bash
pip install scikit-learn xgboost numpy matplotlib seaborn opencv-python-headless tqdm pandas
```

### 2. Kaggle API
- Download your `kaggle.json` from Kaggle → Settings → API Tokens
- Place it at `~/.kaggle/kaggle.json`

### 3. Run notebooks in order
```
1_preprocessing.ipynb       → downloads datasets, saves preprocessed arrays
2_kmeans_segmentation.ipynb → clusters images, extracts features
3_classification.ipynb      → trains and evaluates all models
4_generalization.ipynb      → cross-dataset evaluation on NIH
5_visualizations.ipynb      → generates all report figures
```

All notebooks are designed to run top-to-bottom on **Google Colab (CPU)** with no GPU required.

---

## Results Summary

| Model | Kaggle AUC | NIH AUC | Gap |
|-------|-----------|---------|-----|
| Logistic Regression | — | — | — |
| Random Forest | — | — | — |
| SVM | — | — | — |
| XGBoost (tuned) | — | — | — |

*Results will be filled in after running notebooks.*

---

## Requirements for the setup

- Python 3.8+
- scikit-learn >= 1.0
- xgboost >= 1.6
- numpy, matplotlib, seaborn, opencv-python-headless, tqdm, pandas

---

## Citation

```
@misc{chest_xray_generalization_2024,
  title  = {Chest X-Ray Pneumonia Detection: A Cross-Dataset Generalization Study},
  author = {Aashritha Aachi},
  year   = {20246,
  note   = {CS439 Final Project}
}
```
