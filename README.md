Tourism Experience Analytics
Classification • Rating Prediction • Recommendation System

A full end-to-end Machine Learning project that analyzes tourism behavior to predict user satisfaction, classify visit mode, and generate personalized attraction recommendations.

📌 Project Overview

Tourism agencies and travel platforms aim to improve user experience through data-driven personalization.

This project builds:

⭐ Rating Prediction Model (Regression)

🧭 Visit Mode Prediction (Classification)

🎯 Personalized Recommendation System

🚀 Deployable Streamlit Web Application

The system leverages user behavior, demographic information, and attraction metadata to deliver actionable insights and intelligent recommendations.

🗂 Dataset Description

The dataset consists of structured tourism data across multiple relational tables:

Table	Description
Transaction	User visits & ratings
User	Demographics
City / Country / Region / Continent	Geographic hierarchy
Item	Attraction metadata
Type	Attraction categories
Mode	Visit mode types

Total records:

52,930 transactions

33,530 users

1,698 attractions

🧹 Data Preparation
Cleaning

Removed duplicates

Resolved column conflicts after merges

Standardized date/time fields

Validated rating range (1–5)

Handled missing values

Feature Engineering

Regression Features

VisitQuarter

VisitSeason

Geographic indicators

Classification Features

user_avg_rating

user_rating_count

attraction_avg_rating

attraction_rating_count

Recommendation Features

User–Item interaction matrix

Cosine similarity matrix

📊 Exploratory Data Analysis

Key Insights:

Average rating: 4.16

Most visits from:

United States

Australia

Canada

Most popular attraction types:

Nature & Wildlife

Beaches

Historical Sites

Observation:
User rating behavior is highly consistent — users tend to rate similarly across attractions.

⭐ Regression: Rating Prediction
Models Tested
Model	RMSE	MAE	R²
Linear Regression	0.50	0.29	0.73
Random Forest	0.54	0.27	0.69
Gradient Boosting	0.49	0.27	0.74
✅ Best Model: Gradient Boosting Regressor

R² = 0.74

The model explains 74% of rating variance.

Top Features

user_avg_rating

attraction_avg_rating

user_rating_count

Insight: User behavioral consistency is the strongest predictor of satisfaction.

🧭 Classification: Visit Mode Prediction

Visit Mode Categories:

Business

Family

Couples

Friends

Solo

Initial Training

Accuracy ≈ 47%

After Feature Engineering & Class Balancing
Model	Accuracy
HistGradientBoost	50%
RandomForest (balanced)	48%
Logistic Regression	32%
Interpretation

Multi-class classification (5 classes)

Baseline random guess ≈ 20%

50% accuracy is strong behavioral performance

🎯 Recommendation System
Approach

Collaborative Filtering

Cosine similarity on User–Item matrix

Evaluation

Precision@10 = 0.1369

Meaning:
~13.7% of recommendations matched held-out interactions.

Hybrid Model

Hybrid Precision@10 = 0.1366
No significant improvement over collaborative.

Final choice:
✅ Collaborative Filtering (simpler & stable)
