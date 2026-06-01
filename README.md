# 🏨 Hotel Cancellation Prediction: ML vs. DL Approach

An end-to-end Data Science project focused on predicting hotel booking cancellations. This project implements a comprehensive data engineering pipeline and benchmarks traditional Machine Learning models (Random Forest, XGBoost) against a Deep Learning Multi-Layer Perceptron (MLP) to optimize business revenue.

---

## 📌 Project Overview
In the hospitality industry, booking cancellations significantly impact revenue management and room occupancy planning. This project aims to build a robust predictive system that allows hotels to anticipate cancellations accurately, enabling them to aggressively re-market rooms and minimize financial losses.

## 🛠️ Data Preprocessing & Engineering Pipeline
To ensure high-quality training and prevent data leakage, a strict end-to-end pipeline was implemented:
1. **Feature Selection & Engineering:** Evaluated features meticulously. Dropped highly correlated targets like `reservation_status`, `reservation_status_date`, and `assigned_room_type` to prevent data leakage, and removed irrelevant identifiers.
2. **Handling Missing & Noisy Data:** Cleaned up missing entries and handled anomalies/noisy samples in the dataset.
3. **Categorical Encoding:** Applied proper encoding techniques to transform categorical variables for both tree-based models and neural networks.
4. **Class Imbalance & Scaling:** Analyzed target distribution and applied feature scaling necessary for the Deep Learning model.

---

## 🏆 Model Benchmarking & Performance

We evaluated three different architectures on the test set. Below is the final performance breakdown:

| Model | ROC AUC Score | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost** | **0.918463** | 0.759298 | **0.823756** | **0.790215** |
| **Random Forest** | 0.906951 | 0.745449 | 0.802902 | 0.773109 |
| **Deep Learning (MLP)** | 0.887920 | **0.842885** | 0.638445 | 0.726557 |

### 🔍 Key Insights:
* **XGBoost** is the **Undisputed Champion** for this tabular dataset, achieving the highest **ROC AUC (91.84%)** and **Recall (82.37%)**. High recall is critical for hotels as it flags the maximum number of actual potential cancellations.
* **Deep Learning (MLP)** secured the highest **Precision (84.28%)**, proving that when the neural network predicts a cancellation, it is exceptionally reliable, reducing the risk of false alarms.

---

## 📁 Repository Structure
```text
├── Hotel_Cancellation.ipynb  # Main Jupyter Notebook with full pipeline & models
├── README.md                 # Project documentation
└── requirements.txt          # Dependencies
