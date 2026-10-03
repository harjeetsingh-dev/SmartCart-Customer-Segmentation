import streamlit as st
import pickle
import pandas as pd

# Load trained objects

with open("encoder.pkl", "rb") as f:
    ohe = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)

with open("kmeans.pkl", "rb") as f:
    Kmeans = pickle.load(f)

st.title("SmartCart Customer Segmentation")


#  collect Customer Information  


st.subheader("Customer Information")

income = st.number_input("Income", min_value=0.0)
recency = st.number_input("Recency", min_value=0)
num_deals = st.number_input("NumDealsPurchases", min_value=0)
num_web = st.number_input("NumWebPurchases", min_value=0)
num_catalog = st.number_input("NumCatalogPurchases", min_value=0)
num_store = st.number_input("NumStorePurchases", min_value=0)
num_web_visits = st.number_input("NumWebVisitsMonth", min_value=0)

age = st.number_input("Age", min_value=18)
customer_tenure = st.number_input("Customer Tenure", min_value=0)

t_spending = st.number_input("Total Spending", min_value=0.0)
t_children = st.number_input("Total Children", min_value=0)

education = st.selectbox(
    "Education",
    ["Graduate","Undergraduate", "Postgraduate"]
)

living_with = st.selectbox(
    "Living With",
    ["Alone", "Partner"]
)

complaint = st.selectbox(
    "Complaint",
    [0, 1]
)

response = st.selectbox(
    "Response",
    [0, 1]
)

if st.button("Predict Cluster"):
    
    # Create input Custom DataFrame
    
    input_data = pd.DataFrame({
        "Income": [income],
        "Recency": [recency],
        "NumDealsPurchases": [num_deals],
        "NumWebPurchases": [num_web],
        "NumCatalogPurchases": [num_catalog],
        "NumStorePurchases": [num_store],
        "NumWebVisitsMonth": [num_web_visits],
        "Age": [age],
        "Customer_Tenure": [customer_tenure],
        "T_Spending": [t_spending],
        "T_Children": [t_children],
        "Education": [education],
        "Living_with": [living_with],
        "Complain": [complaint],
        "Response": [response]
    })

    

    # One-Hot Encode on categorical features
    
    cat_cols=["Education","Living_with"]
    
    encoded = ohe.transform(input_data[cat_cols])
    
    encoded_df = pd.DataFrame(
        encoded.toarray(),
        columns=ohe.get_feature_names_out(cat_cols),index=input_data.index)



    #  OneHotEncoder
    #     +
    # Numerical
    

    input_data = pd.concat([input_data.drop(columns=cat_cols),encoded_df],axis=1)


    #  Arrange User input Data as training/model Data frame Order
    
    model_columns = [
    'Income', 'Recency', 'NumDealsPurchases', 'NumWebPurchases',
    'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
    'Complain', 'Response', 'Age', 'Customer_Tenure', 'T_Spending',
    'T_Children', 'Education_Graduate', 'Education_Postgraduate',
    'Education_Undergraduate', 'Living_with_Alone', 'Living_with_Partner'
]
    
    
    input_data = input_data[model_columns]


    # Scale customer Data

    scaled_data = scaler.transform(input_data)


    # PCA Transformation

    X_pca = pca.transform(scaled_data)

    # K-Means prediction
    
    cluster = Kmeans.predict(X_pca)


    st.write("Predicted Cluster:", cluster[0])




    