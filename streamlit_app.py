import streamlit as st

st.title('FutureStaff: Employee Attrition Insights')

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import sklearn
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



rating = ["Very Poor","Poor","Fair","Below Average","Average","Above Average",
           "Good","Very Good","Excellent","Very Excellent"]

# Mapping for MSZoning
businessTravel_mapping = {
    'Non-Travel': 'Non-Travel',
    'Travel Rarely': 'Travel_Rarely',
    'Travel Frequently': 'Travel_Frequently'
}

department_mapping = {
    'Research & Development': 'Research & Development',
    'Sales': 'Sales',
    'Human Resources': 'Human Resources'
}

educationalField_mapping = {
    'Life Sciences': 'Life Sciences',
    'Other': 'Other',
    'Medical': 'Medical',
    'Marketing': 'Marketing',
    'Technical Degree': 'Technical Degree',
    'Human Resources': 'Human Resources',
}

gender_mapping = {
    'Male': 'Male',
    'Female': 'Female'
}

jobRole_mapping = {
    'Research Scientist': 'Research Scientist',
    'Laboratory Technician': 'Laboratory Technician',
    'Manufacturing Director': 'Manufacturing Director',
    'Sales Representative': 'Sales Representative',      
    'Healthcare Representative': 'Healthcare Representative',
    'Research Director': 'Research Director',
    'Sales Executive': 'Sales Executive',
    'Manager': 'Manager',
    'Human Resources': 'Human Resources'
}

maritalStatus_mapping = {
    'Married': 'Married',
    'Single': 'Single',
    'Divorced': 'Divorced'
}

overTime_mapping = {
    'Yes': 'Yes',
    'No': 'No'
}


st.title('Employee Attrition Insights')

st.write('The Best Machine Learning App For Employee Attrition Insights!')
st.write("Develop by: Ivan Lee Kian Hwa,  Kang Quan Pin, Ho Zhi Lin")
# Load the data
with st.expander('Data Set'):
    st.write('**Raw data**')
    df = pd.read_csv('HR-Employee.csv')
    st.write(df)
           
    st.write('**Statistical Summary of Dataset**')
    summary = df.describe().T
    st.write(summary)
           


with st.expander('Data Visualization'):
    st.write('**Scatter Plot**')
    # st.scatter_chart(data=df, x='OverallQual', y='Attrition')  # Modify as needed

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
    with st.expander('Business Travel'):
               businessTravel = st.selectbox('Business Travel', list(businessTravel_mapping.keys()))
               businessTravel_code = businessTravel_mapping[businessTravel]
               st.write("Business travel is: ", businessTravel_code)

    with st.expander('Department'):
               department = st.selectbox('Department', list(department_mapping.keys()))
               department_code = department_mapping[department]
               st.write("Department is: ", department_code)

    with st.expander('Distance From Home'):
               distance = st.number_input("Dzistance", value=None, placeholder="Enter the distance from home")
               st.write("Distance from home is : ", distance ,"KM")

    with st.expander('Education Field'):
               totalBasmtSF = st.number_input("Total square feet of basement area", value=None, placeholder="Enter the square feet")
               st.write("Total square feet of basement area is : ", totalBasmtSF, "sqft")

    with st.expander('Gender'):
               grLiveArea = st.number_input("Above grade (ground) living area square feet", value=None, placeholder="Enter the square feet")
               st.write("Above grade (ground) living area square feet is : ", grLiveArea, "sqft")

    with st.expander('Hourly Rate'):
               fullBath = st.slider("Full bathrooms above grade", 0, 10, 5)
               st.write("Full bathrooms above grade is : ", fullBath)

    with st.expander('Job Role'):
               totalRmsAbvGrd = st.slider("Total rooms above grade (does not include bathrooms)", 1, 20, 10)
               st.write("Total rooms above grade (does not include bathrooms) is : ", totalRmsAbvGrd)

    with st.expander('Marital Status'):
               garageCars = st.slider("Size of garage in car capacity", 0, 10, 3)
               st.write("Size of garage in car capacity is : ", garageCars)

    with st.expander('Monthly Rate'):
               monthlyRate = st.number_input("Monthly Rate", value=None, placeholder="Enter the monthly rate")
               st.write("Monthly rate is : $", monthlyRate)

    # with st.expander('Number Companies Worked'):
    # # Utility input with mapping
    #            utility = st.selectbox('Utility', list(utility_mapping.keys()))
    #            utility_code = utility_mapping[utility]
    #            st.write("Utility code selected is: ", utility_code)

    # with st.expander('Over Time'):
    # # Building Type input with mapping
    #            buildingType = st.selectbox('Building Type', list(buildingType_mapping.keys()))
    #            buildingType_code = buildingType_mapping[buildingType]
    #            st.write("Building Type code selected is: ", buildingType_code)

    with st.expander('Training Times Last Year'): 
    # Map kitchen quality input to corresponding code
               trainingTimesLastYear = st.slider("Training Times Last Year", 0, 5, 3)
               st.write("Training times last year is : ", trainingTimesLastYear)


           
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


# with st.expander('Input Data'):

           # # Ensure input_df has the same structure as df_filtered (used in training)
           # input_df = pd.DataFrame(data, index=[0])
           # st.write('User Input Data')
           # st.write(input_df)
           # input_data = pd.concat([input_df, df_filtered], axis=0)
           
           # important_num_cols.remove("GarageArea")
           # # Handle categorical variables before numeric scaling
           # X = pd.get_dummies(input_data, columns=cat_cols)
           
           
           
           # important_num_cols.remove("SalePrice")
           
           # # Handle the case where the important numeric columns are scaled after dummy encoding
           # # Check if important_num_cols exist in X
           # missing_cols = [col for col in important_num_cols if col not in X.columns]
           
           # if missing_cols:
           #     st.write(f"Warning: The following important numeric columns are missing from the dataset after processing: {missing_cols}")
           
           # # Standardization of data
           # scaler = StandardScaler()
           # # Apply scaler only on numeric columns
           # X[important_num_cols] = scaler.fit_transform(X[important_num_cols])
           # X = X.drop('SalePrice', axis=1)
           
           # # Convert binary columns from 1/0 to True/False
           # for column in X.columns:
           #     if X[column].dtype == 'uint8':  # This is the data type for binary columns created by pd.get_dummies
           #         X = X[column].astype(bool)
           
           # X = X[column_names]
           # st.write('Standardized Input Data')
           # st.write(X[:1])
           
           
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
