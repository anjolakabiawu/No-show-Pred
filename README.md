# 🚗 Balanceè No-Show Prediction

## 🧠 Problem Statement

**Goal:** Predict whether a customer will miss their car service appointment without canceling.

### 💡 Impact
- Reduces lost revenue for service stations  
- Optimizes scheduling and resource allocation  

### ⚠️ Challenge
- Weak predictive signals in current features
- Model performance below random baseline

---

## 🛠️ Approach

### 1. Data Preprocessing

**Handling Missing Values**
```python
df['rating_previous_service'].fillna(df['rating_previous_service'].median(), inplace=True)
```

**Feature Engineering**
- Created interaction terms, e.g.:
  ```python
  df['distance_account_product'] = df['distance_to_station_km'] * df['account_age_days']
  ```
- Binned continuous variables:
  ```python
  df['lead_time_bin'] = pd.cut(df['lead_time_days'], bins=[0, 1, 3, 7, np.inf], labels=['0-1', '1-3', '3-7', '7+'])
  ```

---

### 2. Model Development

| Model              | Hyperparameters                                | Reason for Choice                    |
|-------------------|-------------------------------------------------|--------------------------------------|
| Random Forest      | `n_estimators=100`, `class_weight='balanced'`  | Handles non-linear relationships     |
| Logistic Regression| `max_iter=1000`, `class_weight='balanced'`     | Baseline for linear patterns         |
| XGBoost            | `max_depth=3`, `scale_pos_weight=1`            | Robust to weak features              |

> ✅ **Only Random Forest was used in final submission as it outperformed baseline.**

---

### 3. Evaluation Metrics

**Primary Metric:**  
- F1-score (balances precision and recall)

**Secondary Metrics:**  
- Accuracy  
- Precision / Recall  

---

## 📊 Data Overview

**Key Observations:**
- All features have very low correlation with the target (`< 0.2`)
- PCA shows complete class overlap → No clear separability

---

## 📈 Key Findings

### 1. Model Performance

| Model              | F1-Score | Accuracy | Precision |
|-------------------|----------|----------|---------|
| Random Forest      | 0.534    | 0.525    | 0.544    |
| Logistic Regression| 0.504    | 0.503    | 0.506    |
| XGBoost            | 0.518    | 0.514    | 0.522    |
| **Random Baseline**| **0.561**| **0.536**| **0.489** |

### 2. Critical Insights
- All models performed worse than the **random baseline F1 of 0.561**
- No feature explains >10% of variance (per Random Forest feature importance)
- PCA confirms strong class overlap

---

## 🧨 Root Causes for Low Performance

### 1. Weak Feature Predictive Power
- Max correlation: `0.052` (distance to station)
- PCA clusters show total class mixing

### 2. Potential Data Leakage
- `previous_missed_appointments` might include future data if not carefully time-split

### 3. Lack of Behavioral Data
Missing useful signals like:
- App logins, push notification responses
- Contextual cues (weather, traffic)

---

## 🔄 Next Steps & Recommendations

### 1. Feature Engineering – Add New Data

| Feature Type     | Example Features                        | Data Source       |
|------------------|------------------------------------------|-------------------|
| **User Behavior**| `days_since_last_login`, `reminder_clicks`| App/CRM logs      |
| **Environmental**| `weather_conditions`, `public_holidays`  | Weather API       |
| **Service Context**| `station_peak_hours`, `mechanic_rating`| Internal ops data |

```python
# Example: Integrating weather feature (hypothetical)
df['was_rainy'] = df['appointment_date'].apply(lambda x: check_weather_api(x, 'rain'))
```

---

### 2. Model Improvements

**Unsupervised Learning (to detect user patterns):**
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3).fit(X_scaled)
df['user_segment'] = kmeans.labels_
```

**Deep Learning (if data grows):**
```python
from tensorflow.keras import Sequential, layers

model = Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
```

---

### 3. Business Process Enhancements
- **Deposit System:** Require small deposits for high-risk slots
- **Dynamic Reminders:** Use model predictions to drive SMS/app nudges

---

## 🧪 How to Reproduce

**Install dependencies:**
```bash
pip install pandas scikit-learn xgboost matplotlib seaborn numpy
```

**Run the analysis:**
```bash
python no_show_prediction.py
```

**Expected output:**
```
Model Evaluation:
F1-Score: 0.496
ROC-AUC: 0.52
```

---

📌 **Prepared by:** Anjolaiya Kabiawu  
📅 **Date:** 22 May 2025
