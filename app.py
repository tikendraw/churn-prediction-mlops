from unittest import result
import streamlit as st

st.title('Churn Modelling')
st.image('./Static/quit.webp')

creditscore = st.sidebar.number_input('What is Your Credit score?', 1,1000, value = 420 )
gender = st.sidebar.selectbox('Your Gender',['Male','Female'])
age = st.sidebar.number_input('Your age',10,150, value = 69)
if age > 100:
    if gender =='Male':
        st.sidebar.write('Hello Oldman')
    else:
        st.sidebar.write('Hello Young lady')

tenure = st.sidebar.number_input('Tenure',1,10, value = 2)
balance = st.sidebar.number_input('What is your Bank Balance? ($)',0,5000000, value = 75000)
numberofproducts = st.sidebar.number_input('Number of Product',0,10, value = 2)
estimatedsalary = st.sidebar.number_input('What is Your salary ($)',0, 2000000, value = 55000 )
hascrcard = st.sidebar.selectbox('Do you have Credit card? (yes: 1, no: 0)',[1,0])
geography = st.sidebar.selectbox('Your REGION (pick random if your location is not listed)',['France', 'Spain', 'Germany'])
isactivemember =  st.sidebar.selectbox('Are you an Active Member with the Bank(yes: 1, no: 0)',[1,0])

import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import os
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2' # ignores tf wanrings and red notes


# # load the saved components # scaler and One hot encoder and Model
ohe = joblib.load('./Saved_components/OneHotEncoder.joblib')
scaler = joblib.load('./Saved_components/Scaler.joblib')
new_model = tf.keras.models.load_model('saved_model/my_model')


inp = {
    'Creditscore': creditscore,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumberofProduct': numberofproducts,
    'Hascrcard' : hascrcard,
    'Isactivatemember' : isactivemember,
    'Salary' : estimatedsalary,
    'Geography' : geography,
    'Gender' : gender
}
dd = pd.DataFrame(inp,index=[0])

predict_button = st.button('Predict')

if predict_button:
    st.write('original')
    st.write(dd)

    encoder_df = pd.DataFrame(ohe.transform(dd[['Geography','Gender']]))
    dd = dd.join(encoder_df)
    dd.drop(['Geography', 'Gender'], axis = 1, inplace = True)

    st.write('ohe')
    st.write(dd)

    scaled_dd = scaler.transform(dd)

    st.write('scaled')
    st.write(scaled_dd)
    # result = new_model.predict(scaled_dd)
    # results = np.where(result > 0.5, 1,0)
    # if results == 0:
    #     st.header('No')
    #     st.write(result)
    # else:
    #     st.header('Yes')
    #     st.write(result)

features = '''
**Feature Space** \n
Credit score : Credit score of the customer \n
Geography : The country from which the customer belongs\n
Gender : Male or Female \n
Age : Age of the customer \n
Tenure : Number of years for which the customer has been with the bank \n
Balance : Bank balance of the customer \n
Number of Product : Number of bank products the customer is utilising \n
Has Credit Card :  customer holds a credit card with the bank or not \n
Is Active Member :  customer is an active member with the bank or not \n
Estimated Salary : Estimated salary of the customer in Dollars 
'''
st.write(features)