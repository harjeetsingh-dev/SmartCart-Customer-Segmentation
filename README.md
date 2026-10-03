# SmartCart Customer Segmentation Using Machine Learning

## Project Overview

SmartCart Customer Segmentation is an end-to-end Unsupervised Machine Learning web application designed to segment customers into different groups based on their income, purchasing behavior, spending, engagement, and other customer characteristics.

This project follows a complete Machine Learning workflow including data preprocessing, feature engineering, exploratory data analysis, dimensionality reduction, clustering, model serialization, and deployment using Streamlit.

The trained model analyzes customer information and predicts:

-  Customer Cluster 0
-  Customer Cluster 1
-  Customer Cluster 2
-  Customer Cluster 3

-------- -------- -------- -------- -------- -------- -------- -------- -------- --------

This project follows the complete Machine Learning lifecycle:

- Problem Statement
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Feature Engineering
- Outlier Detection
- Categorical Encoding
- Feature Scaling
- Dimensionality Reduction using PCA
- Customer Clustering
- Model Evaluation
- Model Serialization
- Streamlit Web Application Deployment

-------- -------- -------- -------- -------- -------- -------- --------

#  Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Machine Learning
- K-Means Clustering
- hierarchical clustering
- PCA
- OneHotEncoder
- StandardScaler
  
-------- -------- -------- -------- -------- -------- -------- --------

#  Data Preprocessing

- Handled missing values
- Handled categorical features
- Applied OneHotEncoder
- Applied feature engineering
- Detected and handled outliers
- Feature Scaling using StandardScaler
- Dimensionality Reduction using PCA

-------- -------- -------- -------- -------- -------- -------- --------

#  Dataset Features Used For Customer Segmentation

###  Customer Information

- Income
- Age
- Customer Tenure
- Education
- Living With

###  Purchase Behavior

- NumDealsPurchases
- NumWebPurchases
- NumCatalogPurchases
- NumStorePurchases

###  Customer Engagement

- NumWebVisitsMonth
- Recency

### Spending

- Total Spending

### Customer Family Information

- Total Children

### Customer Response

- Complaint
- Response

-------- -------- -------- -------- -------- -------- -------- --------

#  Machine Learning Algorithm Used

The project uses:

- K-Means Clustering
- hierarchical clustering
- PCA (Principal Component Analysis)

PCA is used for dimensionality reduction before applying K-Means clustering.

The final deployed model:

# K-Means Clustering

The model creates:

**4 Customer Clusters**

-------- -------- -------- -------- -------- -------- -------- --------

# Workflow:

```text
Original Customer Features
          ↓
     StandardScaler
          ↓
          PCA
          ↓
   Reduced Features
          ↓
      K-Means
-------- -------- -------- -------- -------- -------- -------- --------

** live link **

https://smartcart-customer-segmentation-rktd5xtvez9cnm28dp5ggr.streamlit.app/
