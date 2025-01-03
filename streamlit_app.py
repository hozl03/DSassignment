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


st.title('FutureStaff: Employee attrition Insights')

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

loaded_random_forest_classifier = joblib.load('random_forest_classifier.joblib')
loaded_catboost = joblib.load('catboost_model.joblib')
loaded_nn = load_model('neural_network_model.h5')
loaded_random_forest_reg = joblib.load('random_forest_reg.joblib')
loaded_log = joblib.load('log_reg_model.joblib')
loaded_lin = joblib.load('lin_reg_model.joblib')


businessTravel_mapping = {
    'Non-Travel': 0,
    'Travel Rarely': 2,
    'Travel Frequently': 1
}

department_mapping = {
    'Research & Development': 1,
    'Sales': 2,
    'Human Resources': 0
}

educationalField_mapping = {
    'Life Sciences': 1,
    'Other': 4,
    'Medical': 3,
    'Marketing': 2,
    'Technical Degree': 5,
    'Human Resources': 0,
}

gender_mapping = {
    'Male': 1,
    'Female': 0
}

jobRole_mapping = {
    'Research Scientist': 6,
    'Laboratory Technician': 2,
    'Manufacturing Director': 4,
    'Sales Representative': 8,      
    'Healthcare Representative': 0,
    'Research Director': 5,
    'Sales Executive': 7,
    'Manager': 3,
    'Human Resources': 1
}

maritalStatus_mapping = {
    'Married': 1,
    'Single': 2,
    'Divorced': 0
}

overTime_mapping = {
    'Yes': 1,
    'No': 0
}

education_mapping = {
    'Below College': 1,
    'College': 2,
    'Bachelor': 3,
    'Master': 4,
    'Doctor': 5
}

environmentSatisfaction_mapping = {
    'Low': 1,
    'Medium': 2,
    'High': 3,
    'Very High': 4
}

jobInvolvement_mapping = {
    'Low': 1,
    'Medium': 2,
    'High': 3,
    'Very High': 4
}

jobSatisfaction_mapping = {
    'Low': 1,
    'Medium': 2,
    'High': 3,
    'Very High': 4
}

performanceRating_mapping = {
    'Low': 1,
    'Good': 2,
    'Excellent': 3,
    'Outstanding': 4
}

relationshipSatisfaction_mapping = {
    'Low': 1,
    'Medium': 2,
    'High': 3,
    'Very High': 4
}

workLifeBalance_mapping = {
    'Bad': 1,
    'Good': 2,
    'Better': 3,
    'Best': 4
}






st.title('Employee attrition Insights')

st.write('The Best Machine Learning App For Employee attrition Insights!')
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
    df_encoded = df_clean.copy()
    label_encoder = LabelEncoder()

    for col in categorical_cols:
        df_encoded[col] = label_encoder.fit_transform(df_encoded[col])
    st.write(df_encoded.info())
    st.write(df_encoded.head())


           

    # Compute correlation matrix
    correlation_matrix = df_encoded.corr()

    # Get correlation of all features with the target attribute 'attrition'
    attrition_correlation = correlation_matrix['Attrition'].sort_values(ascending=False)


           # Identify columns with correlation <= 0 with 'attrition'
    columns_to_drop = attrition_correlation[attrition_correlation <= 0].index
           
           # Drop these columns from the DataFrame
    df_after_dropping = df_encoded.drop(columns=columns_to_drop)
    df_after_dropping = df_after_dropping.drop(['Over18', 'EmployeeCount', 'StandardHours'], axis=1)
    df_clean = df_after_dropping

           

with st.expander('Data Visualization'):
    # st.write('**Scatter Plot**')
    # st.scatter_chart(data=df, x='OverallQual', y='attrition')  # Modify as needed

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
           
    with st.expander('Age'):
               age = st.number_input("Age", value=None, placeholder="Enter the age")
               st.write("Age is : ", age )

    with st.expander('Business Travel'):
               businessTravel = st.selectbox('Business Travel', list(businessTravel_mapping.keys()))
               businessTravel_code = businessTravel_mapping[businessTravel]
               st.write("Business travel is: ", businessTravel_code)

    with st.expander('Daily Rate'):
               dailyRate = st.number_input("Daily Rate", value=None, placeholder="Enter the daily rate")
               st.write("Daily Rate is : ", dailyRate )

    with st.expander('Department'):
               department = st.selectbox('Department', list(department_mapping.keys()))
               department_code = department_mapping[department]
               st.write("Department is: ", department_code)

    with st.expander('Distance From Home'):
               distance = st.number_input("Distance", value=None, placeholder="Enter the distance from home")
               st.write("Distance from home is : ", distance ,"KM")

    with st.expander('Education'):
               education = st.selectbox('Education', list(education_mapping.keys()))
               education_code = education_mapping[education]
               st.write("Education is: ", education_code)


    with st.expander('Education Field'):
               educationField = st.selectbox('Education Field', list(educationalField_mapping.keys()))
               educationField_code = educationalField_mapping[educationField]
               st.write("Education Field is: ", educationField_code)
    
    with st.expander('Environment Satisfaction'):
               environmentSatisfaction = st.selectbox('Environment Satisfaction', list(environmentSatisfaction_mapping.keys()))
               environmentSatisfaction_code = environmentSatisfaction_mapping[environmentSatisfaction]
               st.write("Environment Satisfaction is: ", environmentSatisfaction_code)


    with st.expander('Gender'):
               gender = st.selectbox('Gender', list(gender_mapping.keys()))
               gender_code = gender_mapping[gender]
               st.write("Gender is: ", gender_code)

    with st.expander('Hourly Rate'):
               hourlyRate = st.number_input("Hourly Rate", value=None, placeholder="Enter the hourly rate")
               st.write("Hourly rate is : ", hourlyRate ,"Hours")

    with st.expander('Job Involvement'):
               jobInvolvement = st.selectbox('Job Involvement', list(jobInvolvement_mapping.keys()))
               jobInvolvement_code = jobInvolvement_mapping[jobInvolvement]
               st.write("Job Involvement is: ", jobInvolvement_code)

    with st.expander('Job Level'):
               jobLevel = st.slider("Job Level", 0, 4, 2)
               st.write("Job level is : ", jobLevel)

    with st.expander('Job Role'):
               jobRole = st.selectbox('Job Role', list(jobRole_mapping.keys()))
               jobRole_code = jobRole_mapping[jobRole]
               st.write("Job role is: ", jobRole_code)
    
    with st.expander('Job Satisfaction'):
               jobSatisfaction = st.selectbox('Job Satisfaction', list(jobSatisfaction_mapping.keys()))
               jobSatisfaction_code = jobSatisfaction_mapping[jobSatisfaction]
               st.write("Environment Satisfaction is: ", jobSatisfaction_code)

    with st.expander('Marital Status'):
               maritalStatus = st.selectbox('Marital Status', list(maritalStatus_mapping.keys()))
               maritalStatus_code = maritalStatus_mapping[maritalStatus]
               st.write("Marital status is: ", maritalStatus_code)

    with st.expander('Monthly Income'):
               monthlyIncome = st.number_input("Monthly Income", value=None, placeholder="Enter the monthly income")
               st.write("Monthly income is : $", monthlyIncome)

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

    with st.expander('Percent Salary Hike'):
               percentSalaryHike = st.number_input("Percent Salary Hike", value=None, placeholder="Enter the Percent Salary Hike")
               st.write("Percent Salary Hike is : ", percentSalaryHike)

    with st.expander('Performance Rating'):
               performanceRating = st.selectbox('Performance Rating', list(performanceRating_mapping.keys()))
               performanceRating_code = performanceRating_mapping[performanceRating]
               st.write("Performance Rating is: ", performanceRating_code)

    with st.expander('Relationship Satisfaction'):
               relationshipSatisfaction = st.selectbox('Relationship Satisfaction', list(relationshipSatisfaction_mapping.keys()))
               relationshipSatisfaction_code = relationshipSatisfaction_mapping[relationshipSatisfaction]
               st.write("Relationship Satisfaction is: ", relationshipSatisfaction_code)

    with st.expander('Stock Option Level'): 
               stockOptionLevel = st.slider("Stock Option Level", 0, 3, 1)
               st.write("Stock Option Level is : ", stockOptionLevel)

    with st.expander('Total Working Years'):
               totalWorkingYears = st.number_input("Total Working Years", value=None, placeholder="Enter the Total Working Years")
               st.write("Total Working Years is : ", totalWorkingYears)

    with st.expander('Training Times Last Year'): 
               trainingTimesLastYear = st.slider("Training Times Last Year", 0, 5, 3)
               st.write("Training times last year is : ", trainingTimesLastYear)

    with st.expander('Work Life Balance'):
               workLifeBalance = st.selectbox('Work Life Balance', list(workLifeBalance_mapping.keys()))
               workLifeBalance_code = workLifeBalance_mapping[workLifeBalance]
               st.write("Work Life Balance is: ", workLifeBalance_code)

    with st.expander('Years At Company'):
               yearsAtCompany = st.number_input("Years At Company", value=None, placeholder="Enter the Years At Company")
               st.write("Years At Company is : ", yearsAtCompany)
               
    with st.expander('Years In Current Role'):
               yearsInCurrentRole = st.number_input("Years In Current Role", value=None, placeholder="Enter the Years In Current Role")
               st.write("Years In Current Role is : ", yearsInCurrentRole)
               
    with st.expander('Years Since Last Promotion'):
               yearsSinceLastPromotion = st.number_input("Years Since Last Promotion", value=None, placeholder="Enter the Years Since Last Promotion")
               st.write("Years Since Last Promotion is : ", yearsSinceLastPromotion)

    with st.expander('Years With Current Manager'):
               yearsWithCurrManager = st.number_input("Years With Current Manager", value=None, placeholder="Enter the Years With Current Manager")
               st.write("Years With Current Manager is : ", yearsWithCurrManager)


           
    # Corrected data dictionary with valid variable names
    data = {
           'Age': age,
           'BusinessTravel': businessTravel_code,
           'DailyRate': dailyRate,
           'Department': department_code,  
           'DistanceFromHome': distance,
           'Education': education_code,
           'EducationField': educationField_code,
           'EnvironmentSatisfaction': environmentSatisfaction_code,
           'Gender': gender_code,  
           'HourlyRate': hourlyRate,
           'JobInvolvement': jobInvolvement_code,
           'JobLevel': jobLevel,
           'JobRole': jobRole_code,
           'JobSatisfaction': jobSatisfaction_code,  
           'MaritalStatus': maritalStatus_code,
           'MonthlyIncome': monthlyIncome,
           'MonthlyRate': monthlyRate,  
           'NumCompaniesWorked': numberCompaniesWorked,  
           'OverTime': overTime_code,
           'PercentSalaryHike': percentSalaryHike,
           'PerformanceRating': performanceRating_code,
           'RelationshipSatisfaction': relationshipSatisfaction_code,
           'StockOptionLevel': stockOptionLevel,
           'TotalWorkingYears': totalWorkingYears,
           'TrainingTimesLastYear': trainingTimesLastYear,
           'WorkLifeBalance': workLifeBalance_code,
           'YearsAtCompany': yearsAtCompany,
           'YearsInCurrentRole': yearsInCurrentRole,
           'YearsSinceLastPromotion': yearsSinceLastPromotion,
           'YearsWithCurrManager': yearsWithCurrManager
    }


           


with st.expander('Input Data'):

           # Ensure input_df has the same structure as df_filtered (used in training)
    input_df = pd.DataFrame(data, index=[0])
    st.write('User Input Data')
    st.write(input_df)
    st.write(input_df.info())
    # input_data = pd.concat([input_df, df_clean], axis=0)

    # # Encode categorical columns
    # categorical_cols_input = input_df.select_dtypes(include=['object']).columns
    # df_encoded_input = input_df.copy()
    # label_encoder = LabelEncoder()

    # for col in categorical_cols_input:
    #     df_encoded_input[col] = label_encoder.fit_transform(df_encoded_input[col])
        # st.write(df_encoded_input[col])
# Split the data into features (X) and target (y)
    X = input_df


# Standardization of numeric data
#     scaler = StandardScaler()

# # Apply scaler only to numeric columns
#     numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
#     X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Now X is ready for input into the model
    st.write('Standardized Input Data')
    st.write(X.head(1))  # Show the standardized data

    # df_encoded.info()
    # df_encoded.head()

           
# Prediction using different models
st.write("## Prediction Results")

# Allow the user to choose the model before pressing Predict
model_choice = st.selectbox('Select Model', ['Random Forest Classifier', 'Random Forest Regression', 'Neural Network', 'Categorical Boosting', 'Logistic Regression', 'Linear Regression'])

if st.button('Predict'):

    if model_choice == 'Neural Network':
        try:
            nn_pred_prob = loaded_nn.predict(X[:1])  # Probability prediction
            nn_pred_class = (nn_pred_prob >= 0.5).astype(int)  # Convert to binary
            st.write(f"**Neural Network Probability: {nn_pred_prob[0][0]:.2f}**")
            st.write(f"**Neural Network Prediction (Class): {nn_pred_class[0]}**")
            st.write("The model predicts a probability of {:.2f}, which is classified as {}.".format(
                nn_pred_prob[0][0],
                "1 (Positive)" if nn_pred_class[0] == 1 else "0 (Negative)"
            ))
        # Additional output based on prediction
            if nn_pred_class[0] == 1:
                st.write("The employee is most likely to attrition.")
            else:
                st.write("The employee is most likely not to attrition.")
        except Exception as e:
            st.error(f"Error during Neural Network prediction: {e}")


    elif model_choice == 'Categorical Boosting':
        try:
            catboost_pred = loaded_catboost.predict(X[:1])
            st.write(f"**Categorical Boosting Prediction: {catboost_pred[0]}**")
            if catboost_pred[0] == 1:
                st.write("The employee is most likely to attrition.")
            else:
                st.write("The employee is most likely not to attrition.")

        except Exception as e:
            st.error(f"Error during Category Boost prediction: {e}")

    elif model_choice == 'Random Forest Classifier':
        try:
            # st.write("Input Data Columns: ", X.columns)
            # st.write("Input shape: ", X[:1].shape)
            random_forest_classifier_pred = loaded_random_forest_classifier.predict(X[:1])
            st.write(f"**Random Forest Classifier Prediction: {random_forest_classifier_pred[0]}**")
            if random_forest_classifier_pred[0] == 1:
                st.write("The employee is most likely to attrition.")
            else:
                st.write("The employee is most likely not to attrition.")
        except Exception as e:
            st.error(f"Error during Random Forest Classifier prediction: {e}")
                   
    elif model_choice == 'Random Forest Regression':
        try:
            random_forest_reg_pred = loaded_random_forest_reg.predict(X[:1])
            random_forest_reg_pred_class = (random_forest_reg_pred >= 0.5).astype(int)  # Convert to binary
            st.write(f"**Random Forest Regression Probability: {random_forest_reg_pred[0]:.2f}**")
            st.write(f"**Random Forest Regression Prediction (Class): {random_forest_reg_pred_class[0]}**")
            st.write("The model predicts a probability of {:.2f}, which is classified as {}.".format(
                random_forest_reg_pred[0],
                "1 (Positive)" if random_forest_reg_pred_class[0] == 1 else "0 (Negative)"
            ))
            if random_forest_reg_pred_class[0] == 1:
                st.write("The employee is most likely to attrition.")
            else:
                st.write("The employee is most likely not to attrition.")
        except Exception as e:
            st.error(f"Error during Random Forest Regression prediction: {e}")

        #     try:
        #     nn_pred_prob = loaded_nn.predict(X[:1])  # Probability prediction
        #     nn_pred_class = (nn_pred_prob >= 0.5).astype(int)  # Convert to binary
        #     st.write(f"**Neural Network Probability: {nn_pred_prob[0][0]:.2f}**")
        #     st.write(f"**Neural Network Prediction (Class): {nn_pred_class[0]}**")
        #     st.write("The model predicts a probability of {:.2f}, which is classified as {}.".format(
        #         nn_pred_prob[0][0],
        #         "1 (Positive)" if nn_pred_class[0] == 1 else "0 (Negative)"
        #     ))
        # # Additional output based on prediction
        #     if nn_pred_class[0] == 1:
        #         st.write("The employee is most likely to attrition.")
        #     else:
        #         st.write("The employee is most likely not to attrition.")
        # except Exception as e:
        #     st.error(f"Error during Neural Network prediction: {e}")


    
    elif model_choice == 'Logistic Regression':
        try:
            log_pred = loaded_log.predict(X[:1])
            st.write(f"**Logistic Regression Prediction: {log_pred[0]}**")
            if log_pred[0] == 1:
                st.write("The employee is most likely to attrition.")
            else:
                st.write("The employee is most likely not to attrition.")
        except Exception as e:
            st.error(f"Error during Logistic Regression prediction: {e}")

    elif model_choice == 'Linear Regression':
        try:
            lin_pred = loaded_lin.predict(X[:1])
            lin_pred_class = (lin_pred >= 0.5).astype(int)  # Convert to binary
            st.write(f"**Linear Regression Probability: {lin_pred[0]}**")
            st.write(f"**Linear Regression Prediction (Class): {lin_pred_class[0]}**")
            st.write("The model predicts a probability of {:.2f}, which is classified as {}.".format(
                lin_pred[0],
                "1 (Positive)" if lin_pred_class[0] == 1 else "0 (Negative)"
            ))

            if lin_pred_class[0] == 1:
                st.write("The employee is most likely to attrition.")
            else:
                st.write("The employee is most likely not to attrition.")
        except Exception as e:
            st.error(f"Error during Linear Regression prediction: {e}")



