# Crop-yield-prediction
Crop Yield Prediction using Machine Learning, NASA POWER climate data, and IMD rainfall datasets.
# Crop Yield Prediction Using Machine Learning

## Overview

This project predicts agricultural crop yield using Machine Learning techniques by integrating crop production data with climate variables such as temperature and rainfall.

The objective is to analyze how environmental factors influence crop productivity and build predictive models that can estimate crop yield for different regions and seasons.

---

## Dataset Sources

### 1. Crop Production Dataset

Source: Kaggle

Dataset contains:

* State
* District
* Crop
* Crop Year
* Season
* Area
* Production
* Yield

### 2. Temperature Data

Temperature data was collected using the NASA POWER API.

Parameter Used:

* T2M (Average Temperature at 2 meters)

Years Covered:

* 1997 – 2020

### 3. Rainfall Data

Rainfall data was obtained from the Indian Meteorological Department (IMD) weather datasets and merged with district-level crop production records.

---

## Final Dataset Features

| Feature     | Description                |
| ----------- | -------------------------- |
| State       | State Name                 |
| District    | District Name              |
| Crop        | Crop Type                  |
| Crop_Year   | Year of Cultivation        |
| Season      | Growing Season             |
| Temperature | Annual Average Temperature |
| Rainfall    | Annual Rainfall            |
| Area        | Cultivated Area            |
| Production  | Crop Production            |
| Yield       | Target Variable            |

---

## Exploratory Data Analysis

The following analyses were performed:

### Crop Frequency Distribution

Visualizes the number of records available for each crop.

### Average Yield Over Years

Shows crop yield trends across different years.

### Top 10 Crops by Average Yield

Identifies crops with the highest average productivity.

### Yield Distribution

Analyzes the distribution of crop yields and detects outliers.

### Outlier Treatment

Extreme yield values were removed using the 2nd and 98th percentile thresholds.

---

## Data Preprocessing

### Numerical Features

* Temperature
* Rainfall
* Area

Applied:

* StandardScaler

### Low Cardinality Categorical Features

* Season
* Crop

Applied:

* One-Hot Encoding

### High Cardinality Categorical Features

* State
* District

Applied:

* Ordinal Encoding

---

## Machine Learning Models

The following regression models were trained and evaluated:

1. Linear Regression
2. Lasso Regression
3. Decision Tree Regressor
4. Random Forest Regressor
5. XGBoost Regressor

---

## Evaluation Metrics

Models were evaluated using:

### Mean Absolute Error (MAE)

Measures average prediction error.

### R² Score

Measures goodness of fit.

---

## Model Comparison

The performance of all models was compared using MAE and R² score.

XGBoost achieved the best performance among all tested algorithms.

---

## Model Saving

The final XGBoost model was saved using Joblib:

```python
joblib.dump(xgb_pipeline, "xgb_crop_yield_model.pkl")
```

Model Loading:

```python
loaded_model = joblib.load("xgb_crop_yield_model.pkl")
```

---

## Project Workflow

1. Data Collection
2. Temperature Data Integration (NASA POWER API)
3. Rainfall Data Integration (IMD)
4. Data Cleaning
5. Exploratory Data Analysis
6. Feature Engineering
7. Data Preprocessing
8. Model Training
9. Model Evaluation
10. Model Deployment Preparation

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Matplotlib
* Seaborn
* Joblib
* NASA POWER API

---

## Future Improvements

* Include soil characteristics
* Include humidity and wind speed
* Hyperparameter tuning using GridSearchCV
* Deploy as a web application using Streamlit
* Real-time weather integration

---

## Results

The project successfully demonstrates that climate variables such as temperature and rainfall significantly contribute to crop yield prediction.

Among all tested models, XGBoost provided the most accurate predictions and can be further optimized for deployment in agricultural decision-support systems.

---

## Author

Lohith SC

Machine Learning and Data Science Enthusiast

