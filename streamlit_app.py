import streamlit as st

st.title('HEHEHE')

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import sklearn # import the module so you can use it
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures

import joblib

loaded_random_forest = joblib.load('random_forest_model.joblib')
loaded_svr = joblib.load('svr_model.joblib')
loaded_lin_reg = joblib.load('linear_regression_model.joblib')

column_names = joblib.load('saved_feature_names.pkl')
scaler = joblib.load('scaler.pkl')


rating = ["Very Poor","Poor","Fair","Below Average","Average","Above Average",
           "Good","Very Good","Excellent","Very Excellent"]

# Mapping for MSZoning
msZoning_mapping = {
    'Commercial': 'C (all)',
    'Floating Village Residential': 'FV',
    'Residential High Density': 'RH',
    'Residential Low Density': 'RL',
    'Residential Medium Density': 'RM'
}

utility_mapping = {
    'All Public Utilities': 'AllPub',
    'Electricity, Gas, and Water (Septic Tank)': 'NoSewr',
}

landSlope_mapping = {
    'Gentle slope': 'Gtl',
    'Moderate Slope': 'Mod',
    'Severe Slope': 'Sev'
}

buildingType_mapping = {
    'Single-family Detached': '1Fam',
    'Two-family Conversion': '2FmCon',
    'Duplex': 'Duplx',
    'Townhouse End Unit': 'TwnhsE',
    'Townhouse Inside Unit': 'TwnhsI'
}

kitchenQual_mapping = {
    'Excellent': 'Ex',
    'Good': 'Gd',
    'Average': 'TA',
    'Fair': 'Fa',
}

saleCondition_mapping = {
    'Normal Sale': 'Normal',
    'Abnormal Sale': 'Abnorml',
    'Adjoining Land Purchase': 'AdjLand',
    'Allocation': 'Alloca',
    'Family': 'Family',
    'Partial': 'Partial'
}


st.title('House Price Prediction')

st.write('The Best Machine Learning App For House Price Prediction!')
st.write("Develop by: Tay Keng Aik,  Cheong Wai Kian, Ho Zhi Lin")
# Load the data
with st.expander('Data Set'):
    st.write('**Raw data**')
    df = pd.read_csv('train.csv')
    st.write(df)
           
    st.write('**Statistical Summary of Dataset**')
    summary = df.describe().T
    st.write(summary)
           

    # Select more important columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Numeric columns only
    if 'SalePrice' in numeric_df.columns:
        important_num_cols = list(numeric_df.corr()["SalePrice"][
            (numeric_df.corr()["SalePrice"] > 0.50) | (numeric_df.corr()["SalePrice"] < -0.50)
        ].index)
    else:
        important_num_cols = []

    # Categorical columns
    cat_cols = ["MSZoning", "Utilities", "BldgType", "KitchenQual", "SaleCondition", "LandSlope"]
    important_cols = important_num_cols + cat_cols

    df_filtered = df[important_cols]
    st.write("**Filtered Data with Important Columns**")

    df_filtered = df_filtered.drop('GarageArea', axis=1)
    st.write(df_filtered)


with st.expander('Data Visualization'):
    st.write('**Scatter Plot**')
    st.scatter_chart(data=df, x='OverallQual', y='SalePrice')  # Modify as needed

    st.write('**Correlation Heatmap**')
    
    # Filter out non-numeric columns for the heatmap
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if not numeric_df.empty:
        # Generate heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), cmap="RdBu", ax=ax)
        ax.set_title("Correlations Between Variables", size=15)
        st.pyplot(fig)
    else:
        st.write("No numeric columns available for correlation heatmap.")





# Input
with st.sidebar:

    st.header('Input features')
    with st.expander('Overall Quality'):
               overallQuality = st.slider("Rates the overall material and finish of the house", 1, 10, 7)
               st.write("The overall material and finish of the house is : ", rating[overallQuality - 1])

    with st.expander('Year Built'):
               yearBuilt = st.number_input("Original construction year", value=None, placeholder="Enter a year")
               st.write("The original construction year is: ", yearBuilt)

    with st.expander('Remodel Date'):
               yearRemodAdd = st.number_input("Remodel year", value=None, placeholder="Enter a year")
               st.write("The remodel year is: ", yearBuilt)

    with st.expander('Total Basement Area'):
               totalBasmtSF = st.number_input("Total square feet of basement area", value=None, placeholder="Enter the square feet")
               st.write("Total square feet of basement area is : ", totalBasmtSF, "sqft")

    with st.expander('First Floor Square Feet'):
               floorSF = st.number_input("First Floor square feet", value=None, placeholder="Enter the square feet")
               st.write("First Floor square feet is : ", floorSF, "sqft")

    with st.expander('Above Grade Living Area'):
               grLiveArea = st.number_input("Above grade (ground) living area square feet", value=None, placeholder="Enter the square feet")
               st.write("Above grade (ground) living area square feet is : ", grLiveArea, "sqft")

    with st.expander('Full Bathrooms Above Grade'):
               fullBath = st.slider("Full bathrooms above grade", 0, 10, 5)
               st.write("Full bathrooms above grade is : ", fullBath)

    with st.expander('Total Rooms Above Grade'):
               totalRmsAbvGrd = st.slider("Total rooms above grade (does not include bathrooms)", 1, 20, 10)
               st.write("Total rooms above grade (does not include bathrooms) is : ", totalRmsAbvGrd)

    with st.expander('Size of Garage'):
               garageCars = st.slider("Size of garage in car capacity", 0, 10, 3)
               st.write("Size of garage in car capacity is : ", garageCars)

    with st.expander('Zoning'):
               msZoning = st.selectbox('Zoning', list(msZoning_mapping.keys()))
               msZoning_code = msZoning_mapping[msZoning]  # Map to the corresponding code (e.g., "A", "C")
               st.write("Zoning code selected is: ", msZoning_code)

    with st.expander('Utility'):
    # Utility input with mapping
               utility = st.selectbox('Utility', list(utility_mapping.keys()))
               utility_code = utility_mapping[utility]
               st.write("Utility code selected is: ", utility_code)

    with st.expander('Building Type'):
    # Building Type input with mapping
               buildingType = st.selectbox('Building Type', list(buildingType_mapping.keys()))
               buildingType_code = buildingType_mapping[buildingType]
               st.write("Building Type code selected is: ", buildingType_code)

    with st.expander('Kitchen Quality'): 
    # Map kitchen quality input to corresponding code
               kitchenQual = st.selectbox('Kitchen Quality', list(kitchenQual_mapping.keys()))
               kitchenQual_code = kitchenQual_mapping[kitchenQual]  # Map to the corresponding code (e.g., "Ex", "Gd")
               st.write("Kitchen Quality code selected is: ", kitchenQual_code)

    with st.expander('Sale Condition'):
    # Sale Condition input with mapping
               saleCondition = st.selectbox('Condition of Sale', list(saleCondition_mapping.keys()))
               saleCondition_code = saleCondition_mapping[saleCondition]
               st.write("Sale Condition code selected is: ", saleCondition_code)

    with st.expander('Land Slope'):
    # Land Slope input with mapping
               landSlope = st.selectbox('Land Slope', list(landSlope_mapping.keys()))
               landSlope_code = landSlope_mapping[landSlope]
               st.write("Land Slope code selected is: ", landSlope_code)


           
    # Corrected data dictionary with valid variable names
    data = {
           'OverallQual': overallQuality,  # Use overallQuality from slider
           'YearBuilt': yearBuilt,  # Extract the year from date input
           'YearRemodAdd': yearRemodAdd,  # Extract the year from date input
           'TotalBsmtSF': totalBasmtSF,  # Use totalBasmtSF from slider
           'TotRmsAbvGrd': totalRmsAbvGrd,  # Use totalRmsAbvGrd from slider
           '1stFlrSF': floorSF,  # Use floorSF from slider
           'GrLivArea': grLiveArea,  # Use grLiveArea from slider
           'FullBath': fullBath,  # Use fullBath from slider
           'GarageCars': garageCars,  # Use garageCars from slider
           'MSZoning': msZoning_code,  # Use msZoning_code
           'Utilities': utility_code,  # Use utility_code
           'BldgType': buildingType_code,  # Use buildingType_code
           'KitchenQual': kitchenQual_code,  # Use kitchenQual_code from selectbox
           'SaleCondition': saleCondition_code,  # Use saleCondition_code from selectbox
           'LandSlope': landSlope_code,  # Use landSlope_code

           # 'OverallQual': 5,  # Use overallQuality from slider
           # 'YearBuilt': 1958,  # Extract the year from date input
           # 'YearRemodAdd': 1985,  # Extract the year from date input
           # 'TotalBsmtSF': 912,  # Use totalBasmtSF from slider
           # 'TotRmsAbvGrd': 5,  # Use totalRmsAbvGrd from slider
           # '1stFlrSF': 912,  # Use floorSF from slider
           # 'GrLivArea': 912,  # Use grLiveArea from slider
           # 'FullBath': 1,  # Use fullBath from slider
           # 'GarageCars': 1,  # Use garageCars from slider
           # 'MSZoning': "RL",  # Use msZoning_code
           # 'Utilities': "AllPub",  # Use utility_code
           # 'BldgType': "1Fam",  # Use buildingType_code
           # 'KitchenQual': "TA",  # Use kitchenQual_code from selectbox
           # 'SaleCondition': "Normal",  # Use saleCondition_code from selectbox
           # 'LandSlope': "Gtl",  # Use landSlope_code

    }


with st.expander('Input Data'):

           # Ensure input_df has the same structure as df_filtered (used in training)
           input_df = pd.DataFrame(data, index=[0])
           st.write('User Input Data')
           st.write(input_df)
           input_data = pd.concat([input_df, df_filtered], axis=0)
           
           important_num_cols.remove("GarageArea")
           # Handle categorical variables before numeric scaling
           X = pd.get_dummies(input_data, columns=cat_cols)
           
           
           
           important_num_cols.remove("SalePrice")
           
           # Handle the case where the important numeric columns are scaled after dummy encoding
           # Check if important_num_cols exist in X
           missing_cols = [col for col in important_num_cols if col not in X.columns]
           
           if missing_cols:
               st.write(f"Warning: The following important numeric columns are missing from the dataset after processing: {missing_cols}")
           
           # Standardization of data
           scaler = StandardScaler()
           # Apply scaler only on numeric columns
           X[important_num_cols] = scaler.fit_transform(X[important_num_cols])
           X = X.drop('SalePrice', axis=1)
           
           # Convert binary columns from 1/0 to True/False
           for column in X.columns:
               if X[column].dtype == 'uint8':  # This is the data type for binary columns created by pd.get_dummies
                   X = X[column].astype(bool)
           
           X = X[column_names]
           st.write('Standardized Input Data')
           st.write(X[:1])
           
           
# Prediction using different models
st.write("## Prediction Results")

# Allow the user to choose the model before pressing Predict
model_choice = st.selectbox('Select Model', ['Random Forest', 'Support Vector Regression', 'Linear Regression'])

if st.button('Predict'):
    # Check for missing inputs (NaN or None values)
    missing_values = input_df.isnull().sum()

    # Check if there are any missing values in the user's input
    if missing_values.any():
        st.error(f"Please fill out all the required fields. Missing values: {list(input_df.columns[missing_values > 0])}")
    else:
        # Proceed with prediction only if no values are missing
        if model_choice == 'Linear Regression':
            lin_reg_pred = loaded_lin_reg.predict(X)
            st.write(f"**Linear Regression Prediction: ${lin_reg_pred[0]:,.2f}**")

        elif model_choice == 'Support Vector Regression':
            svr_pred = loaded_svr.predict(X)
            st.write(f"**SVR (GridSearch) Prediction: ${svr_pred[0]:,.2f}**")

        elif model_choice == 'Random Forest':
            random_forest_pred = loaded_random_forest.predict(X)
            st.write(f"**Random Forest Prediction: ${random_forest_pred[0]:,.2f}**")
