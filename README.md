# Kubernetes Cluster Issue Prediction

## Project Overview
This project focuses on detecting and predicting potential failures in Kubernetes clusters using machine learning techniques. The primary goal is to enhance Kubernetes reliability by identifying anomalies and predicting issues based on network traffic, CPU/memory usage, and other resource metrics.

## Dataset
The dataset consists of network traffic metrics, container resource usage, and various other Kubernetes-specific features. It includes fields such as:
- CPU and memory usage statistics
- Network traffic details (bytes sent/received, packets, etc.)
- Flow-based features (duration, protocol, timestamps)
- Various statistical measures (mean, standard deviation of resource consumption)
- **Label Column**: Indicating whether an issue occurred

## Problem Statement
Kubernetes clusters can face various issues, such as:
- **Node/Pod Failures**
- **Resource Exhaustion (CPU/Memory Overload)**
- **Network Traffic Bottlenecks**
- **Service Disruptions**

This project builds a machine learning model to predict these failures using classification and anomaly detection techniques.

---

## Methodology
### 1. Data Preprocessing
- Load the dataset and check for missing values.
- Fill missing values using median imputation for numerical columns.
- Feature selection and engineering:
  - Creating ratio-based features (e.g., network bytes per packet)
  - Aggregate metrics (sum of packets, bytes across different time intervals)
  - Log transformations to handle skewed data

### 2. Model Training
- **Supervised Learning (Classification Model)**
  - Random Forest Classifier trained on labeled data
  - Evaluated using accuracy, precision, recall, F1-score
- **Unsupervised Learning (Anomaly Detection)**
  - Isolation Forest for detecting anomalies based on resource usage patterns
  - Used to flag outliers that may indicate potential failures

### 3. Model Evaluation
- Confusion Matrix and classification report
- Visualization of false positives/negatives
- Comparison of model performance with baseline approaches

### 4. Model Deployment
- Trained models are saved using `joblib` for later use.
- Predictions can be made on new data using both the classification model and anomaly detection model.

---

## Installation & Usage
### Prerequisites
Ensure you have the following dependencies installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Running the Model
The script will:
   - Load and preprocess the dataset
   - Train the models
   - Evaluate performance
   - Save the models for later use

### Using the Trained Model for Prediction
Once trained, the model can be used for predictions:
```python
import joblib
import pandas as pd

# Load models
classifier_model = joblib.load('kubernetes_classifier_model.pkl')
anomaly_model = joblib.load('kubernetes_anomaly_model.pkl')
feature_names = joblib.load('model_feature_names.pkl')

# Load new data and ensure it has the same features
new_data = pd.read_csv('new_data.csv')[feature_names]
predictions = classifier_model.predict(new_data)
print(predictions)
```

---

## Results & Key Insights
- **Classification Model:** Achieved high accuracy in predicting known issues based on historical data.
- **Anomaly Detection:** Successfully flagged previously unseen potential failures using Isolation Forest.
- **Feature Importance Analysis:** Key indicators of failures include high CPU/memory usage, sudden spikes in network traffic, and extreme variations in container resource limits.
- **Confusion Matrix Observations:**
  - False negatives indicate cases where the model did not detect an issue despite real failures.
  - False positives show cases where the model predicted issues that did not occur.

---

## Future Improvements
- Incorporate real-time monitoring and alert systems.
- Improve feature engineering by incorporating domain-specific Kubernetes insights.
- Experiment with deep learning models for time-series anomaly detection.

---

## Authors
Developed for Guidewire's AI-driven Kubernetes hackathon. Contributions include data collection, feature engineering, model training, evaluation, and deployment strategies.

For further improvements or contributions, feel free to submit issues or pull requests!

