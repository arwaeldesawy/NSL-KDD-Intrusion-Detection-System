# NSL-KDD Intrusion Detection System

## Overview
This project is an **Intrusion Detection System (IDS)** using the **NSL-KDD dataset**. It explores network traffic data, visualizes patterns, and applies Machine Learning models to classify attacks.

The project includes:
- **Data Exploration:** Inspect dataset, check for missing values.
- **Data Visualization:** Explore distributions, correlations, and trends.
- **Machine Learning Models:** Compare SVM, KNN, and Random Forest with accuracy and confusion matrices.

---

## Dataset
- **Source:** NSL-KDD dataset
- **Files:**
  - `NSL_KDD_Train.csv` - training data
  - `NSL_KDD_Test.csv` - testing data
- **Columns:** 42 features including numeric and categorical features like `protocol_type`, `service`, `flag`, etc.
- **Target:** `label` column (Normal / Attack)

---

## Features
1. `duration`, `protocol_type`, `service`, `flag`, `src_bytes`, `dst_bytes`, etc.
2. Binary target label created as:
   - `0` → Normal
   - `1` → Attack

---

## Machine Learning Models
- **SVM**
  - Accuracy: 99.1%
  - Works well with scaled data, strong generalization
- **KNN**
  - Accuracy: 98.7%
  - Simple, but sensitive to scaling and noise

---

## How to Run
1. Install required packages:  
```bash
pip install -r requirements.txt
