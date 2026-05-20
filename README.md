# Medical ML Pipeline

End-to-end machine learning pipeline applied to medical data — covering both **supervised image classification** (brain MRI tumor detection) and **unsupervised text mining** (clustering of patient symptom descriptions).

This repository showcases a complete ML workflow: data cleaning, feature engineering, model training, comparative evaluation, and result interpretation.

---

## Overview

| Notebook | Type | Task |
|---|---|---|
| `supervised_preprocessing.ipynb` | Supervised | Clean, normalize and split brain MRI images into train / val / test sets |
| `supervised_training.ipynb` | Supervised | Train and compare 5 classifiers to detect brain tumors |
| `unsupervised_learning.ipynb` | Unsupervised | Discover topics in free-text medical symptom descriptions via K-Means clustering |

---

## 1. Supervised Pipeline — Brain Tumor Classification

**Goal:** classify brain MRI scans into 4 categories — *glioma*, *meningioma*, *pituitary*, *no tumor*.

### Preprocessing (`supervised_preprocessing.ipynb`)
- Load images from a 4-class directory structure
- Validate file integrity, remove corrupt samples
- Convert to grayscale, apply mild contrast enhancement
- Resize to **224×224** and normalize to `[0, 1]`
- Stratified split: **72% train / 8% validation / 20% test**
- Export to `brain_tumor_cleaned.npz`

### Training (`supervised_training.ipynb`)
- Flatten images (50,176 features) and reduce with **PCA** (95% variance → 725 components)
- Standardize features with `StandardScaler`
- Train and compare **5 models** with diverse loss functions:
  - Logistic Regression — cross-entropy
  - K-Nearest Neighbors — distance-based
  - SVM (RBF kernel) — hinge loss
  - Random Forest — Gini impurity
  - Gradient Boosting — boosted decision trees
- Evaluate with **accuracy, precision, recall, F1-score**, and confusion matrices

### Result
**Gradient Boosting** achieved the best performance:
> **Test F1-score = 0.8221**

---

## 2. Unsupervised Pipeline — Symptom Clustering

**Goal:** group similar patient symptom descriptions without using any labels.

### Approach (`unsupervised_learning.ipynb`)
- Tokenize text and remove stop words
- Vectorize with **TF-IDF** (top 2,000 features)
- Reduce dimensionality with **Truncated SVD / LSA** (200 components, ≥95% variance retained)
- L2-normalize the embeddings
- Cluster with **K-Means**, with optimal *K* selected via **Silhouette Score**
- Evaluate cluster quality: Silhouette, Davies-Bouldin, Calinski-Harabasz
- Visualize clusters using **PCA** and **t-SNE**
- Export cluster assignments to `results_clusters.csv`

---

## Dataset

`Student_Dataset.csv` is included **as an example only**. It contains free-text descriptions of medical symptoms and injuries used to demonstrate the unsupervised pipeline. Real-world deployment would use a larger and properly curated clinical corpus.

The brain MRI images used in the supervised pipeline are not redistributed here — the notebook expects them in a standard 4-class directory layout.

---

## Tech Stack

- **Python 3**
- **Data:** `pandas`, `numpy`, `Pillow`
- **ML:** `scikit-learn` (PCA, SVD, K-Means, classifiers, metrics)
- **Visualization:** `matplotlib`, `seaborn`, `t-SNE`
- **Environment:** Jupyter Notebook

---

## Getting Started

```bash
# Clone the repository
git clone <repo-url>
cd medical-ml-pipeline

# Install dependencies
pip install numpy pandas scikit-learn pillow matplotlib seaborn jupyter

# Launch Jupyter
jupyter notebook
```

Then open any of the three notebooks and run the cells top to bottom.

---

## Project Structure

```
medical-ml-pipeline/
├── supervised_preprocessing.ipynb   # Image cleaning & dataset preparation
├── supervised_training.ipynb        # Model training & evaluation
├── unsupervised_learning.ipynb      # Text clustering pipeline
├── Student_Dataset.csv              # Example dataset (symptom descriptions)
└── README.md
```

---

## Key Takeaways

- **Full ML workflow:** raw data → preprocessing → feature engineering → training → evaluation → interpretation
- **Both paradigms covered:** supervised image classification *and* unsupervised text mining
- **Rigorous evaluation:** multiple models, multiple metrics, train/val/test discipline
- **Reproducible:** stratified splits, fixed pipelines, saved intermediate artifacts

