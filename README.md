# 💳 Credit Card Fraud Detection — 99.75% Accuracy

> Detecting fraudulent credit card transactions using Unsupervised Machine Learning techniques: **Isolation Forest** and **Local Outlier Factor (LOF)**.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Approach](#approach)
- [Algorithms](#algorithms)
  - [Isolation Forest](#1-isolation-forest)
  - [Local Outlier Factor](#2-local-outlier-factor)
- [Implementation](#implementation)
- [Results](#results)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)
- [Key Takeaways](#key-takeaways)

---

## Overview

Credit card fraud detection is the process of identifying and rejecting fraudulent purchase attempts before they are processed. This project applies **Unsupervised Machine Learning** anomaly detection algorithms to flag transactions that deviate significantly from normal behavior — without relying on labeled training data in a traditional supervised sense.

The notebook achieves a remarkable **~99.75% accuracy**, demonstrating that anomaly detection is a powerful tool for highly imbalanced fraud datasets.

---

## Problem Statement

Credit card companies must detect fraudulent transactions to:

- Prevent unauthorized charges to customers
- Reduce financial losses for banks and merchants
- Maintain customer trust and security

The core challenge is that **fraudulent transactions are extremely rare** compared to legitimate ones (heavily imbalanced dataset), making traditional classification approaches less effective without careful tuning.

---

## Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **File:** `creditcard.csv`
- **Features:** Anonymized PCA-transformed features (`V1`–`V28`), plus `Time` and `Amount`
- **Target Column:** `Class`
  - `0` → Normal Transaction
  - `1` → Fraudulent Transaction

### Class Distribution

The dataset is **highly imbalanced**, with the vast majority of transactions being legitimate. The contamination ratio (fraud / normal) is calculated programmatically and passed directly into both models.

### Exploratory Visualization

A histogram comparing transaction **amounts** between fraud and normal classes is plotted (log-scaled Y-axis) to surface behavioral differences between the two groups.

---

## Approach

Rather than supervised learning (which requires large amounts of labeled fraud examples), this notebook uses **unsupervised anomaly detection**:

1. The models learn what "normal" looks like from the data distribution.
2. Transactions that deviate significantly from normal are flagged as anomalies (fraud).
3. The `contamination` parameter tells each model the expected proportion of outliers.

```python
contamination = len(df[df['Class'] == 1]) / float(len(df[df['Class'] == 0]))
```

---

## Algorithms

### 1. Isolation Forest

**Library:** `sklearn.ensemble.IsolationForest`

#### How It Works

Isolation Forest isolates observations by:
1. Randomly selecting a feature.
2. Randomly selecting a split value between the feature's min and max.
3. Repeating until the observation is isolated (forms a leaf node).

**Key Insight:** Anomalies (fraud) are isolated in **fewer splits** than normal transactions, because they occupy sparse, extreme regions of feature space. A lower average path length = higher anomaly score.

#### Parameters Used

| Parameter      | Value   | Description                                              |
|----------------|---------|----------------------------------------------------------|
| `n_estimators` | 1000    | Number of trees in the forest (more = more stable)       |
| `max_samples`  | `"auto"`| Samples per tree (defaults to `min(256, n_samples)`)     |
| `contamination`| Computed| Proportion of outliers expected in the dataset           |
| `random_state` | 19      | Seed for reproducibility                                 |

#### Output Mapping

The model outputs `1` (normal) and `-1` (anomaly). These are remapped to match the dataset's `Class` convention:

```python
df['Class_IF'] = abs(df['Class_IF'] - 1) // 2
# 1 (normal)  → 0
# -1 (fraud)  → 1
```

---

### 2. Local Outlier Factor

**Library:** `sklearn.neighbors.LocalOutlierFactor`

#### How It Works

LOF compares the **local density** of a data point to its neighbors using Euclidean distance:

1. Computes the reachability distance of each point to its `k` nearest neighbors.
2. Calculates the Local Reachability Density (LRD) — a measure of how dense the region around a point is.
3. Compares a point's LRD to its neighbors'. A low relative density = likely anomaly.

**Key Insight:** Fraudulent transactions cluster in low-density regions, surrounded by sparse neighbors, while normal transactions cluster tightly.

#### Parameters Used

| Parameter       | Value   | Description                                                      |
|-----------------|---------|------------------------------------------------------------------|
| `n_neighbors`   | 50      | Number of neighbors for density estimation                        |
| `leaf_size`     | 10      | Size of the BallTree/KDTree leaf nodes (affects speed/accuracy)  |
| `contamination` | Computed| Expected proportion of outliers                                  |

#### Output Mapping

Same remapping as Isolation Forest:

```python
df['Class_LOF'] = abs(df['Class_LOF'] - 1) // 2
```

---

## Implementation

### Full Pipeline

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score
from tabulate import tabulate

# Load data
df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")

# Features (exclude target column)
columnNames = list(df.columns)[:-1]

# Compute contamination ratio
contamination = len(df[df['Class'] == 1]) / float(len(df[df['Class'] == 0]))

# --- Isolation Forest ---
cIF = IsolationForest(n_estimators=1000, max_samples="auto",
                      contamination=contamination, random_state=19)
df['Class_IF'] = cIF.fit_predict(df[columnNames].values)
df['Class_IF'] = abs(df['Class_IF'] - 1) // 2

# --- Local Outlier Factor ---
cLOF = LocalOutlierFactor(n_neighbors=50, leaf_size=10, contamination=contamination)
df['Class_LOF'] = cLOF.fit_predict(df[columnNames].values)
df['Class_LOF'] = abs(df['Class_LOF'] - 1) // 2
```

### Evaluation

```python
data = [
    ["Isolation Forest",     accuracy_score(df['Class'], df['Class_IF'])],
    ["Local Outlier Factor", accuracy_score(df['Class'], df['Class_LOF'])]
]
print(tabulate(data, headers=["Algorithm", "Accuracy"], tablefmt="fancy_grid"))
```

---

## Results

| Algorithm            | Accuracy    |
|----------------------|-------------|
| Isolation Forest     | ~99.75%     |
| Local Outlier Factor | ~99.65%     |

> ⚠️ **Note on accuracy with imbalanced data:** Because fraud is very rare (~0.17% of transactions), even a naive model that predicts everything as "normal" would achieve ~99.83% accuracy. For production use, consider also evaluating **Precision, Recall, F1-Score, and ROC-AUC** — especially recall on the fraud class, which directly measures how many fraudulent transactions are actually caught.

---

## Dependencies

Install all required packages via pip:

```bash
pip install numpy pandas matplotlib scikit-learn tabulate
```

| Library       | Purpose                                        |
|---------------|------------------------------------------------|
| `numpy`       | Numerical operations                           |
| `pandas`      | Data loading and manipulation                  |
| `matplotlib`  | Data visualization (histograms)                |
| `scikit-learn`| Isolation Forest, LOF, accuracy scoring        |
| `tabulate`    | Pretty-printing the results comparison table   |

---

## How to Run

### On Kaggle (Recommended)
1. Open the notebook in [Kaggle Notebooks](https://www.kaggle.com/).
2. Add the **Credit Card Fraud Detection** dataset to your notebook inputs.
3. Run all cells in order.

### Locally
1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Update the file path in the notebook:
   ```python
   df = pd.read_csv("path/to/creditcard.csv")
   ```
3. Install dependencies and run the notebook with Jupyter.

---

## Key Takeaways

- **Unsupervised methods work well** for fraud detection when labeled fraud samples are scarce.
- **Isolation Forest** is generally faster and scales better to large datasets.
- **Local Outlier Factor** can be more precise in dense, locally structured data but is slower due to neighbor computation.
- The `contamination` parameter is critical — setting it accurately (from the actual class ratio) significantly improves both models.
- For real-world deployment, accuracy alone is insufficient; **recall on the fraud class** is the metric that matters most operationally.

---

## 📂 Project Structure

```
├── credit-card-fraud-detection-99-75-accuracy.ipynb   # Main notebook
├── creditcard.csv                                      # Dataset (download from Kaggle)
└── README.md                                           # This file
```

---

*Built with ❤️ using scikit-learn's anomaly detection toolkit.*
