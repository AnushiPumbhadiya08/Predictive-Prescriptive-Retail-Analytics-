# Predictive-Prescriptive-Retail-Analytics-
A predictive and prescriptive analytics project that models consumer behavior for retail using machine learning and dashboarding. Includes coupon usage prediction, regional and seasonal inventory optimization, customer satisfaction analysis, and revenue driver modeling with a Grafana visualization layer.

## Overview 
This project analyzes consumer behavior in retail using a combination of machine learning, descriptive analytics, and dashboard visualization. The goal is to enable data-driven decision making across marketing, inventory management, customer satisfaction, and revenue optimization.

The analysis integrates:

- Predictive modeling

- Product demand ranking

- Feature importance analysis

- Regional & seasonal insights

- Grafana dashboard visualization

## Problem Statement 
Modern retailers collect large volumes of transaction and behavior data, yet struggle to convert it into actionable decisions. The project addresses four key business questions:

1. Who is most likely to use coupons?

2. Which products should each region/season prioritize in inventory?

3. What factors drive customer satisfaction (ratings)?

4. What drives purchase value (order amount)?

## Dataset Description
Kaggle link: https://www.kaggle.com/datasets/zeesolver/consumer-behavior-and-shopping-habits-dataset 

The dataset contains 3,900 retail transactions across 50 US locations, including demographic, behavioral, product, logistics, and marketing variables.

**Dataset Dimensions (from dashboard)**
| Metric                | Value       |
| --------------------- | ----------- |
| Total Transactions    | **3900**    |
| Total Revenue (USD)   | **233,081** |
| Avg Order Value (USD) | **59.8**    |
| Unique Locations      | **50**      |
| Unique Seasons        | **4**       |
| Unique Items          | **25**      |

Note: The Kaggle dataset originally used in this project is no longer publicly accessible on the 
Kaggle platform. All analysis in this report are based on the version downloaded prior to its 
removal, and the dataset cannot currently be retrieved or referenced through the original Kaggle 
link 

## Data Preprocessing
Data cleaning and preprocessing steps included:

- Handling missing values

- Converting numeric fields

- Feature construction

- Encoding categorical features

- Standardization for ML models

- Fairness adjustment

## Models Implemented
**1. Coupon Usage Prediction (Classification)**
- Model: Logistic Regression

- Purpose: Identify users likely to redeem coupons

- Pipeline: StandardScaler + OneHotEncoder + LogisticRegression

- Result:

  - Accuracy: 0.83

  - AUC: 0.81

  - Threshold strategy used for selective coupon allocation

**2. Inventory Optimization (Prescriptive Ranking)**
- Method: Hierarchical aggregation

- Grouping: Season → Location → Item

- Metrics: Sales count + Total revenue

- Outputs:

  - Global top items

  - Seasonal/Regional priority items

**3. Satisfaction Analysis (Review Rating)**
- Model: Random Forest Regressor

- Objective: Identify which attributes influence review ratings

- Performance:

  - RMSE: 0.76

  - R²: -0.05 (predictive accuracy low → interpretation only)

**4. Revenue Driver Analysis (Order Value Prediction)**
- Models: Random Forest, Gradient Boosting

- Performance (Best Model):

  - RMSE: 23.82

  - R²: -0.01

## Dashboard Visualization
A Grafana dashboard was built for real-time visualization of:

- Core KPIs

- Revenue distribution

- Seasonal sales heatmaps

- Geospatial revenue clusters

- Product ranking

## Technologies Used
- Python

- Pandas / NumPy

- Scikit-learn

- Random Forest / Gradient Boosting

- Grafana

- CSV-based Data Storage

## Key Findings
- Coupons should be targeted toward subscribers and apparel buyers

- Inventory should follow a two-layer strategy:

  - Global stable core items

  - Local seasonal boosters

- Satisfaction improvements depend on logistics + regional quality

- Revenue segmentation should use LTV + product behavior, not discounts

## Limitations
- Dataset lacks time-series behavior

- Limited logistics & service metrics

- Gender imbalance required removal

- Regression tasks lacked explanatory signals

- Dashboard collaboration limited by user permissions

## Future Improvements
- Add return behavior, shipping delays, baskets

- Integrate causal modeling for ratings

- Reinforcement learning for promo allocation

- Real-time streaming pipeline

- ERP integration for planning & stock control

## Conclusion
This project demonstrates how machine learning and descriptive analytics can support data-driven retail strategy across marketing, operations, and customer experience. The combination of predictive modeling and dashboarding enables both interpretation and actionability.

## Contributors
Anushi Pumbhadiya (github.com/AnushiPumbhadiya08)

Arjun Parikh (github.com/ArjunParikh1404)
