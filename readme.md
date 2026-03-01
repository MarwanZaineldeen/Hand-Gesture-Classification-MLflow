# 🖐️ NeuralGesture
### 🚀 Advanced Hand Gesture Recognition & End-to-End MLOps Pipeline

An end-to-end **Machine Learning research project** focused on classifying **18 distinct hand gestures** using **MediaPipe landmarks**.

This repository demonstrates a complete **MLOps lifecycle**, featuring:

- 🔬 Multi-model benchmarking  
- ⚙️ Hyperparameter research & optimization  
- 📊 Automated experiment tracking with MLflow  
- 🚀 Real-time inference deployment  

---

## 📌 Project Overview

NeuralGesture transforms raw hand movements into structured numerical representations using 3D landmark extraction, then evaluates multiple ML architectures to identify the optimal balance between:

- 🎯 Classification Accuracy  
- ⚡ Inference Speed  
- 🔁 Model Generalization  
- 📦 Production Readiness  

---

## 🏗️ Project Architecture & File Structure

```text
📂 NeuralGesture
├── 📁 mlruns/                           # MLflow local tracking database
├── 📁 ML flow UI screenshots/           # Experiment tracking evidence
├── 📜 hand_gesture_classification.ipynb # Core research notebook
├── 📜 mlflow_logger.py                  # Custom automated MLflow logger
├── 📜 live_demo.py                      # Real-time inference script
├── 📜 hand_landmarks_data.csv           # Processed landmark dataset
├── 📜 label_encoder.pkl                 # Serialized gesture encoders
├── 📜 requirements.txt                  # Project dependencies
└── 📜 README.md

```
---


## 🧪 Technical Implementation

### 1️⃣ Feature Extraction & Engineering

- ✋ Extracted **21 high-precision 3D landmarks** per hand  
- 📐 Generated **63-dimensional feature vectors**  
- 🔄 Applied normalization for spatial invariance  
- 🧠 Transformed physical gestures into ML-ready structured inputs  

---

### 2️⃣ Multi-Model Benchmarking

The following models were evaluated:

- 🌲 XGBoost  
- 🌳 Random Forest  
- 📍 K-Nearest Neighbors  
- 🌿 Decision Tree  
- 📈 Support Vector Machine (Baseline + Optimized)  
- ⚡ AdaBoost  
- 📊 Logistic Regression  

Each model was assessed on:

- 🎯 Accuracy  
- 📊 Per-class recall  
- 🔍 Confusion matrix analysis  
- ⚡ Inference efficiency  

---

## 🏆 Final Results & Model Leaderboard

After extensive benchmarking, **XGBoost** emerged as the Champion 🏆

| Rank | Model | Accuracy | Status |
|------|--------|----------|--------|
| 🥇 | XGBoost | 97.99% | Champion 🏆 |
| 🥈 | Random Forest (Optimized) | 97.88% | Runner-up |
| 🥉 | KNN | 97.72% | Strong Baseline |
| 4️⃣ | Decision Tree | 95.07% | High Interpretability |
| 5️⃣ | SVM (Baseline) | 93.34% | Reliable |
| 6️⃣ | SVM (Poly Kernel) | 88.57% | Research Variant |
| 7️⃣ | AdaBoost | 85.82% | - |
| 8️⃣ | Logistic Regression | 85.67% | Linear Baseline |

---

## 📊 MLOps Integration

### 🗂️ Dataset Lineage Tracking

Every MLflow run tagged with:

- 📌 Dataset version  
- ⚙️ Model parameters  
- 📈 Metrics  

Ensuring **100% reproducibility**.

---

### 🧾 Automated Artifact Generation

Custom logging engine (`mlflow_logger.py`) automatically generates:

- 📄 `per_class_metrics.csv`  
- 📊 Confusion matrices  
- 📦 Model artifacts and run metadata  

---

### 📈 Performance Dashboards

Each experiment produces a **Master Performance Dashboard** including:

- 📊 Confusion Matrix  
- 🍩 Accuracy visualization  
- 📉 Per-gesture recall analysis  

---

## 🚀 Real-Time Inference

Run the live demo:

```bash
python live_demo.py

```

### Features

- 🎥 Real-time hand tracking  
- ⚡ Instant gesture classification  
- 🏷️ Serialized label decoding  
- 🧩 Production-style inference pipeline  

---

## 🔧 Installation & Reproducibility

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/NeuralGesture.git
cd NeuralGesture
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Launch MLflow UI

```bash
mlflow ui
```

Open in browser:

```text
http://127.0.0.1:5000
```

---

## 🧰 Technology Stack

- 🐍 Python  
- 🤖 Scikit-learn  
- 🌲 XGBoost  
- ✋ MediaPipe  
- 📦 MLflow  
- 🎥 OpenCV  
- 📊 Pandas  
- 🔢 NumPy  

---

## 🎯 Key Achievements

- 🏆 97.99% classification accuracy  
- 📊 8-model benchmarking framework  
- 🔁 Automated MLflow experiment tracking  
- 📦 Dataset lineage reproducibility  
- 🚀 Real-time inference deployment  
- 🧱 Production-ready modular architecture  

---

## 🎓 Acknowledgments

This research was conducted as part of the **Information Technology Institute (ITI)** — 9-Month Artificial Intelligence Program.

---