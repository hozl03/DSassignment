import random
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import log_loss
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split

import streamlit as st


st.title('FutureStaff: Employee Attrition Insights')

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime
import catboost
from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

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
from tensorflow.keras.models import load_model

loaded_random_forest = joblib.load('random_forest_model.joblib')
loaded_catboost = joblib.load('catboost_model.joblib')
loaded_nn = load_model('neural_network_model.h5')



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


    # Detecting outliers using IQR (Interquartile Range)
    def detect_outliers(df, column):
      Q1 = df[column].quantile(0.25)
      Q3 = df[column].quantile(0.75)
      IQR = Q3 - Q1
      lower_bound = Q1 - 1.5 * IQR
      upper_bound = Q3 + 1.5 * IQR
      outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
      return outliers, lower_bound, upper_bound

#To detect outliers using IQR, we can only use numerical field for detection.
#We have come up with a set of columns to detect outliers as shown in below
#Detect outliers for selected columns in the dataset
    columns_to_check = ['Age', 'DailyRate', 'DistanceFromHome', 'MonthlyIncome',
                    'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike',
                    'TotalWorkingYears', 'TrainingTimesLastYear',
                    'YearsAtCompany', 'YearsInCurrentRole',
                    'YearsSinceLastPromotion', 'YearsWithCurrManager']

    outliers_summary = {}

    for column in columns_to_check:
      outliers, lower_bound, upper_bound = detect_outliers(df, column)
      outliers_summary[column] = [outliers.shape[0], lower_bound, upper_bound]

# Convert the summary into a DataFrame for better readability
    outliers_summary_df = pd.DataFrame(outliers_summary,
                                  index=['Number of Outliers', 'Lower Bound', 'Upper Bound'])

    outliers_summary_df


# Filter columns with outliers
    columns_with_outliers = [column for column, values in outliers_summary.items() if values[0] > 0]

# Drop columns without outliers from the original dataset
    df_filtered = df[columns_with_outliers]

# Remove outliers for the retained columns
    df_clean = df.copy()  # Start with a copy of the original dataset
    for column in columns_with_outliers:
        lb = outliers_summary[column][1]  # Lower Bound
        ub = outliers_summary[column][2]  # Upper Bound
        df_clean = df_clean[(df_clean[column] >= lb) & (df_clean[column] <= ub)]

# Number of rows before and after outlier removal
    rows_before = df.shape[0]
    rows_after = df_clean.shape[0]
    rows_removed = rows_before - rows_after


# Check for unrealistic or negative values
    def detect_negative_values(df, columns_to_check):
        negative_values_summary = {}
        for column in columns_to_check:
            negative_count = df[df[column] < 0].shape[0]
            negative_values_summary[column] = negative_count
        return negative_values_summary

# Specify columns to check for negative or unrealistic values
    columns_to_check = ['Age', 'DailyRate', 'MonthlyIncome', 'DistanceFromHome',
                    'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole',
                    'YearsSinceLastPromotion', 'YearsWithCurrManager']

# Detect negative values
    negative_values_summary = detect_negative_values(df_clean, columns_to_check)

# Convert the summary to a DataFrame for better readability
    negative_values_df = pd.DataFrame.from_dict(
        negative_values_summary, orient='index', columns=['Negative Value Count']
    )




           
    # Encode categorical columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    encoded_df = df_clean.copy()
    label_encoder = LabelEncoder()
           
    for col in categorical_cols:
           encoded_df[col] = label_encoder.fit_transform(encoded_df[col])


    # Compute correlation matrix
    correlation_matrix = encoded_df.corr()

    # Get correlation of all features with the target attribute 'Attrition'
    attrition_correlation = correlation_matrix['Attrition'].sort_values(ascending=False)


           # Identify columns with correlation <= 0 with 'Attrition'
    columns_to_drop = attrition_correlation[attrition_correlation <= 0].index
           
           # Drop these columns from the DataFrame
    df_after_dropping = encoded_df.drop(columns=columns_to_drop)
    df_after_dropping = df_after_dropping.drop(['Over18', 'EmployeeCount', 'StandardHours'], axis=1)
    df_clean = df_after_dropping

           

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
               distance = st.number_input("Distance", value=None, placeholder="Enter the distance from home")
               st.write("Distance from home is : ", distance ,"KM")

    with st.expander('Education Field'):
               educationField = st.selectbox('educationField', list(educationalField_mapping.keys()))
               educationField_code = educationalField_mapping[educationField]
               st.write("Education Field is: ", educationField_code)

    with st.expander('Gender'):
               gender = st.selectbox('Gender', list(gender_mapping.keys()))
               gender_code = gender_mapping[gender]
               st.write("Gender is: ", gender_code)

    with st.expander('Hourly Rate'):
               hourlyRate = st.number_input("Hourly Rate", value=None, placeholder="Enter the hourly rate")
               st.write("Hourly rate is : ", hourlyRate ,"Hours")

    with st.expander('Job Role'):
               jobRole = st.selectbox('Job Role', list(jobRole_mapping.keys()))
               jobRole_code = jobRole_mapping[jobRole]
               st.write("Job role is: ", jobRole_code)

    with st.expander('Marital Status'):
               maritalStatus = st.selectbox('Marital Status', list(maritalStatus_mapping.keys()))
               maritalStatus_code = maritalStatus_mapping[maritalStatus]
               st.write("Marital status is: ", maritalStatus_code)

    with st.expander('Monthly Rate'):
               monthlyRate = st.number_input("Monthly Rate", value=None, placeholder="Enter the monthly rate")
               st.write("Monthly rate is : $", monthlyRate)

    with st.expander('Number Companies Worked'):
               numberCompaniesWorked = st.slider("Number Companies Worked", 0, 10, 5)
               st.write("Number companies worked is : ", numberCompaniesWorked)

    with st.expander('Over Time'):
               overTime = st.selectbox('Over Time', list(overTime_mapping.keys()))
               overTime_code = overTime_mapping[overTime]
               st.write("Over Time is: ", overTime_code)

    with st.expander('Training Times Last Year'): 
               trainingTimesLastYear = st.slider("Training Times Last Year", 0, 5, 3)
               st.write("Training times last year is : ", trainingTimesLastYear)
   
           
    # Corrected data dictionary with valid variable names
    data = {
           'BusinessTravel': businessTravel,  
           'Department': department,  
           'DistanceFromHome': distance,  
           'EducationField': educationField,  
           'Gender': gender,  
           'HourlyRate': hourlyRate,  
           'JobRole': jobRole,  
           'MaritalStatus': maritalStatus,  
           'MonthlyRate': monthlyRate,  
           'NumCompaniesWorked': numberCompaniesWorked,  
           'OverTime': overTime,  
           'TrainingTimesLastYear': trainingTimesLastYear
    }


           


with st.expander('Input Data'):

           # Ensure input_df has the same structure as df_filtered (used in training)
    input_df = pd.DataFrame(data, index=[0])
    st.write('User Input Data')
    st.write(input_df)
    input_data = pd.concat([input_df, df_clean], axis=0)
           
# Handle categorical variables before numeric scaling
    categorical_col = []  # Initialize list

    for column in df_clean.columns:
# Check if the column is of object type or category type and has limited unique values
        if (df_clean[column].dtype == 'object' or df_clean[column].dtype.name == 'category') and len(df_clean[column].unique()) <= 50:
            categorical_col.append(column)

    df_clean['Attrition'] = df_clean.Attrition.astype("category").cat.codes

# Check if 'Attrition' is in the list before removing
    if "Attrition" in categorical_col:
# Transform categorical data into dummies
        categorical_col.remove("Attrition")
        data = pd.get_dummies(df, columns=categorical_col)
    else:
# Handle the case where 'Attrition' is not in the list
        data = df.copy()




# Handle the case where the important numeric columns are scaled after dummy encoding
# # Check if important_num_cols exist in X
#     missing_cols = [col for col in important_num_cols if col not in X.columns]

#     if missing_cols:
#         st.write(f"Warning: The following important numeric columns are missing from the dataset after processing: {missing_cols}")

# # Standardization of data
#     scaler = StandardScaler()
# # Apply scaler only on numeric columns
#     X[df_clean] = scaler.fit_transform(X[df_clean])
#     X = X.drop('Attrition', axis=1)

# # Convert binary columns from 1/0 to True/False
#     for column in X.columns:
#         if X[column].dtype == 'uint8':  # This is the data type for binary columns created by pd.get_dummies
#             X = X[column].astype(bool)

#     X = X[column_names]
#     st.write('Standardized Input Data')
#     st.write(X[:1])

# Split the data into features (X) and target (y)
    X = df_clean.drop(columns=['Attrition'])  # Drop the target column
    y = df_clean['Attrition']

# Standardization of numeric data
    scaler = StandardScaler()

# Apply scaler only to numeric columns
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Now X is ready for input into the model
    st.write('Standardized Input Data')
    st.write(X.head(1))  # Show the standardized data

           
# Prediction using different models
st.write("## Prediction Results")

# Allow the user to choose the model before pressing Predict
model_choice = st.selectbox('Select Model', ['Random Forest Classifier', 'Support Vector Classifier', 'Neural Network', 'Category Boost'])

if st.button('Predict'):
    # Check if the model_choice is valid and prediction can proceed
    if model_choice == 'Neural Network':
        nn_pred_prob = loaded_nn.predict(X[:1])  # Probability prediction
        nn_pred_class = (nn_pred_prob >= 0.5).astype(int)  # Convert to binary
        st.write(f"**Neural Network Probability: {nn_pred_prob[0][0]:.2f}**")
        st.write(f"**Neural Network Prediction (Class): {nn_pred_class[0]}**")
        st.write("The model predicts a probability of {:.2f}, which is classified as {}.".format(
            nn_pred_prob[0][0],
            "1 (Positive)" if nn_pred_class[0] == 1 else "0 (Negative)"
        ))

    elif model_choice == 'Category Boost':
        try:
            catboost_pred = loaded_catboost.predict(X[:1])
            st.write(f"**Category Boost Prediction: {catboost_pred[0]}**")
        except Exception as e:
            st.error(f"Error during Category Boost prediction: {e}")

    elif model_choice == 'Random Forest Classifier':
        try:
            random_forest_pred = loaded_random_forest.predict(X[:1])
            st.write(f"**Random Forest Prediction: {random_forest_pred[0]}**")
        except Exception as e:
            st.error(f"Error during Random Forest prediction: {e}")

    # if model_choice == 'Category Boost':
    #     catboost_pred = loaded_catboost.predict(X)
    #     st.write(f"**Category Boost Prediction: {catboost_pred[0]:,.2f}**")


    # Check for missing inputs (NaN or None values)
    # missing_values = isnull().sum()

    # Check if there are any missing values in the user's input
    # if missing_values.any():
    #     st.error(f"Please fill out all the required fields. Missing values: {list(input_df.columns[missing_values > 0])}")
    # else:
        # Proceed with prediction only if no values are missing
        # if model_choice == 'Neural Network':
        #     nn_pred = loaded_nn.predict(X)
        #     st.write(f"**Neural Network Prediction: {nn_pred[0]:,.2f}**")

        # elif model_choice == 'Category Boost':
        #     catboost_pred = loaded_catboost.predict(X)
        #     st.write(f"**Category Boost Prediction: {catboost_pred[0]:,.2f}**")

        # elif model_choice == 'Random Forest':
        #     random_forest_pred = loaded_random_forest.predict(X)
        #     st.write(f"**Random Forest Prediction: {random_forest_pred[0]:}**")
