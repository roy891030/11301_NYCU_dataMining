# 11301_NYCU_dataMining

This repository contains my data mining class homeworks for the 11301 semester at NYCU. Each folder corresponds to a specific homework, showcasing a variety of data analysis and machine learning techniques.

## Table of Contents
- [1_game_of_thrones](#1_game_of_thrones)
- [2_Sentiment_Analysis](#2_sentiment_analysis)
- [3_Time_Series_Regression](#3_time_series_regression)
- [4_Sentiment_Analysis_2](#4_sentiment_analysis_2)
- [5_movieRating](#5_movierating)

---

## 1_game_of_thrones
### Overview
Predict the survival of Game of Thrones characters using various machine learning techniques, including Decision Trees and XGBoost.

### Key Highlights
- **Preprocessing:** Handling missing values and creating dummy variables.
- **Advanced Features:** Incorporating battle risk as a feature for prediction.
- **Model:** Decision Tree and XGBoost, tuned with cross-validation.

**Path:** `1_game_of_thrones/`

---

## 2_Sentiment_Analysis
### Overview
Perform sentiment analysis on Amazon reviews using TF-IDF and Word2Vec embeddings, evaluated with Random Forest and XGBoost classifiers.

### Key Highlights
- **Preprocessing:** Text tokenization and stop-word removal.
- **Feature Extraction:** TF-IDF and Word2Vec embeddings.
- **Evaluation:** Cross-validation accuracy compared with Kaggle leaderboard scores.

**Path:** `2_Sentiment_Analysis/`

---

## 3_Time_Series_Regression
### Overview
Time series regression for predicting air pollution (PM2.5) levels using features from historical data.

### Key Highlights
- **Preprocessing:** Missing value imputation and time window creation.
- **Models:** Linear Regression and XGBoost.
- **Evaluation:** Metrics include Mean Absolute Error (MAE) for 1-hour and 6-hour predictions.

**Path:** `3_Time_Series_Regression/`

---

## 4_Sentiment_Analysis_2
### Overview
Deep learning-based sentiment analysis of Yelp reviews using CNN and LSTM models.

### Key Highlights
- **Models:** Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM).
- **Results:** Comparison of training and validation accuracy for both models.
- **Evaluation:** Confusion matrix analysis.

**Path:** `4_Sentiment_Analysis_2/`

---

## 5_movieRating
### Overview
Predict user movie ratings using matrix factorization with embedding layers.

### Key Highlights
- **Preprocessing:** Randomized dataset splitting into training and testing sets.
- **Model:** Collaborative filtering using user and movie embeddings.
- **Evaluation:** Mean Absolute Error (MAE) on predictions.

**Path:** `5_movieRating/`

---

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/roy891030/11301_NYCU_dataMining.git
   cd 11301_NYCU_dataMining
