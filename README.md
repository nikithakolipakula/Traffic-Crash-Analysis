# 🚦 Traffic Crash Analysis & Prediction

Comprehensive data analysis of traffic crash patterns using Python, data visualization, and machine learning.

---

## 📋 Overview

This project analyzes traffic crash data to identify patterns between speed limits, crash timing, road conditions, and injury severity. The analysis combines exploratory data analysis (EDA), statistical methods, and predictive modeling to uncover meaningful insights for road safety and accident prevention.

Dataset: Traffic_Crashes_-_Crashes.csv
Language: Python
Focus: Data-driven traffic safety and crash prediction

---

## 🚀 Quick Start

### Install Dependencies

pip install pandas numpy matplotlib seaborn plotly scikit-learn streamlit

### Run Application

streamlit run app.py

The dashboard will open in your browser with interactive visualizations and prediction features.

---

## 📂 Project Files

app.py                          # Main Streamlit dashboard
new_code.py                     # Updated analysis + ML logic
code.py                         # Initial version of analysis
traffic_cleaned.csv             # Cleaned dataset
Traffic_Crashes_-_Crashes.csv   # Raw dataset
README.md                       # Documentation

---

## 🔍 Analysis Components

### 🧹 Data Processing

Cleaning: Remove duplicates, handle missing values
Imputation: Fill missing values using mean/median
Transformation: Convert date/time into usable features
Filtering: Remove irrelevant or noisy data
Outliers: Detection using statistical methods

---

### 📊 Exploratory Data Analysis

Comprehensive visualizations and insights:

Distributions → Speed limits, crash frequency
Relationships → Speed vs injuries (scatter plots)
Correlations → Heatmap of numeric features
Trends → Crash occurrences by hour/day
Comparisons → High-risk zones and conditions
Patterns → Injury severity vs speed

---

### 📈 Statistical Methods

Descriptive statistics (mean, std deviation)
Correlation analysis
Trend analysis by time
Outlier detection
Distribution analysis

---

## 🤖 Machine Learning Model

### Linear Regression

Predict injuries based on:

* Speed
* Number of lanes
* Crash hour

Performance: Continuous prediction of injury severity

---

## 📊 Dataset Schema

Column	Type	Description

POSTED_SPEED_LIMIT	Integer	Road speed limit
LANE_CNT	Integer	Number of lanes
CRASH_HOUR	Integer	Time of crash (0–23)
INJURIES_TOTAL	Integer	Total injuries in crash
CRASH_DATE	Date	Date of accident
LOCATION	String	Crash location details

---

## 🔄 Analysis Pipeline

Load Data
↓
Clean & Preprocess
↓
Exploratory Analysis (Visualizations)
↓
Statistical Analysis
↓
Feature Selection
↓
Model Training
↓
Prediction & Insights

---

## 📌 Key Deliverables

✓ Interactive Streamlit dashboard
✓ Real-time filters (speed, hour, sample size)
✓ Dynamic visualizations
✓ Correlation heatmap
✓ Crash trend analysis
✓ Speed vs injury relationship
✓ Predictive injury model
✓ Clean and optimized dataset

---

## ⚙️ Data Handling Strategy

Type	Missing Values	Strategy

Numeric	NaN	Fill with mean
Categorical	Missing	Replace with "Unknown"
Duplicates	All	Remove
Outliers	Extreme values	Handled using filtering

---

## 📦 Requirements

pandas >= 1.0.0
numpy >= 1.18.0
matplotlib >= 3.1.0
seaborn >= 0.10.0
plotly >= 5.0.0
scikit-learn >= 0.22.0
streamlit >= 1.0.0

---

## 💻 Output

Running the app provides:

Dashboard: Interactive visual interface
Visualizations: Charts, heatmaps, scatter plots
Metrics: Crash statistics and averages
Prediction: Injury prediction based on inputs
Insights: Trends and patterns in crash data

---

## 📚 File Guide

app.py

Lines 1-20: Imports & setup
Lines 21-60: Data loading & cleaning
Lines 61-120: Dashboard filters & KPIs
Lines 121-200: Visualizations
Lines 201+: Machine learning model

---

## 👤 Author

Course: INT 375
Project: Traffic Crash Analysis & Prediction

---

## 📄 License

Academic coursework project

---

## 💡 Note

This project demonstrates how data analysis and machine learning can be applied to real-world traffic data to improve safety and decision-making.

### 🚀 Future Improvements
- Add real-time traffic prediction
