# 11301_NYCU_dataMining

This repository contains my data mining assignments for the 11301 semester at NYCU. Each assignment focuses on different data analysis techniques, including sentiment analysis, time series regression, and recommendation systems.

## Table of Contents
- [1_game_of_thrones](#1_game-of-thrones)
- [2_Sentiment_Analysis](#2-sentiment-analysis)
- [3_Time_Series_Regression](#3-time-series-regression)
- [4_Sentiment_Analysis_2](#4-sentiment-analysis-2)
- [5_movieRating](#5-movierating)

---

## 1_game_of_thrones
### Overview
This project predicts the likelihood of a character's survival in the Game of Thrones universe using classification models.

### Methodology
- **Data Preprocessing:** Handling missing values, converting "Death Year" to binary data, and creating dummy variables for categorical features like allegiances.
- **Feature Engineering:** Incorporating battle risk scores for each house based on historical data.
- **Modeling:** Decision Tree classification and XGBoost with hyperparameter tuning.

### Results
- **Decision Tree Accuracy:** 65.12%
- **XGBoost Cross-validation Score:** 72.3%
- **Final Kaggle Submission Score:** 73.16%

**Path:** `1_game_of_thrones/`

---

## 2_Sentiment_Analysis
### Overview
This project performs sentiment analysis on Amazon reviews using both TF-IDF and Word2Vec embeddings with classification models.

### Methodology
- **Data Preprocessing:** Stop word removal, text tokenization, and converting scores to binary sentiment (positive/negative).
- **Feature Extraction:** Using TF-IDF vectorization and Word2Vec embeddings to represent text data.
- **Model Comparison:** Evaluating Random Forest and XGBoost models with k-fold cross-validation.

### Results
| Model | Cross-validation Accuracy | Kaggle Score |
|--------|--------------------------|--------------|
| Random Forest (TF-IDF) | 80.04% | 62.64% |
| XGBoost (TF-IDF) | 82.62% | 62.57% |
| Random Forest (Word2Vec) | 76.62% | 55.97% |
| XGBoost (Word2Vec) | 76.89% | 71.52% |

**Path:** `2_Sentiment_Analysis/`

---

## 3_Time_Series_Regression
### Overview
This project predicts PM2.5 air pollution levels using time series regression models.

### Methodology
- **Data Preprocessing:** Missing value imputation with hourly averages, normalization, and splitting data into training (October/November) and testing (December) sets.
- **Feature Engineering:** Creating time-windowed features for 1-hour and 6-hour predictions.
- **Modeling:** Comparing Linear Regression and XGBoost models on single and multi-feature datasets.

### Results
| Model | 1-hour Prediction MAE | 6-hour Prediction MAE |
|--------|----------------------|----------------------|
| Linear Regression (PM2.5 only) | 2.67 | 4.31 |
| Linear Regression (All Features) | 2.65 | 4.27 |
| XGBoost (PM2.5 only) | 2.89 | 4.47 |
| XGBoost (All Features) | 2.81 | 4.62 |

**Path:** `3_Time_Series_Regression/`

---

## 4_Sentiment_Analysis_2
### Overview
This project builds deep learning models (CNN and LSTM) for sentiment classification on Yelp reviews.

### Methodology
- **Data Preprocessing:** Stop word removal, text tokenization, and TF-IDF vectorization.
- **CNN Model:** Includes convolutional layers, max pooling, dropout, and dense layers.
- **LSTM Model:** Bidirectional LSTM layers with batch normalization, dense layers, and dropout for sequence modeling.
- **Model Evaluation:** Training and validation accuracy, loss, and confusion matrix analysis.

### Results
| Model | Training Accuracy | Validation Accuracy |
|--------|------------------|------------------|
| CNN | 87% | 75% |
| LSTM | 74% | 74% |

**Path:** `4_Sentiment_Analysis_2/`

---

## 5_movieRating
### Overview
This project implements a collaborative filtering model using matrix factorization to predict user movie ratings.

### Methodology
- **Data Preprocessing:** Randomized dataset splitting into training (80%) and testing (20%) sets.
- **Matrix Factorization:** Using embedding layers for users and movies, combined with dot product and bias terms.
- **Training:** Optimized with Adam optimizer and mean squared error loss.
- **Evaluation:** Using Mean Absolute Error (MAE) to measure prediction performance.

### Results
| Model | MAE |
|--------|------|
| Base Matrix Factorization (50D Embeddings, 5 epochs) | 0.719 |
| Enhanced Matrix Factorization (100D Embeddings, 10 epochs) | 0.697 |

**Path:** `5_movieRating/`

---

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/roy891030/11301_NYCU_dataMining.git
   cd 11301_NYCU_dataMining
   ```
2. Navigate to the desired homework folder and explore the code.
3. Ensure you have the necessary Python libraries installed. You can use:
   ```bash
   pip install -r requirements.txt
   ```

## Contact
If you have any questions or feedback, feel free to reach out!

📧 Email: [roy60404@gmail.com](roy60404@gmail.com)
📌 GitHub: [roy891030](https://github.com/roy891030)

