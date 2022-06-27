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
from sklearn.compose import ColumnTransformer
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2' # ignores tf wanrings and red notes


# # load the saved components # scaler and One hot encoder and Model
ohe = joblib.load('./saved_components/ohe.joblib')
scaler = joblib.load('./saved_components/scaler.joblib')
coltrans2 = joblib.load('./saved_components/coltransformer.joblib')
new_model = tf.keras.models.load_model('saved_model/my_model.h5')

inp = {
    'CreditScore': creditscore,
    'Geography' : geography,
    'Gender' : gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': numberofproducts,
    'HasCrCard' : hascrcard,
    'IsActiveMember' : isactivemember,
    'EstimatedSalary' : estimatedsalary
}

#Creating DAtaframe from user input
X = pd.DataFrame(inp,index=[0])

# defining columns
cat_cols = ['Geography', 'Gender']
num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
        'IsActiveMember', 'EstimatedSalary']

# Predict button
predict_button = st.button('Predict')

if predict_button:
    st.write('original')
    st.write(X)

    
    # coltrans = ColumnTransformer([('scaler', scaler, num_cols),
    #                          ('ohencoder', ohe, cat_cols)],remainder= 'passthrough')
    feature_space = coltrans2.transform(X)
    st.write('scaled and encoded')
    st.write(feature_space)
    result = new_model.predict(feature_space)
    results = np.where(result > 0.5, 1,0)
    if results == 0:
        # st.header('Stayed')
        st.markdown(f'<p style="color:#E44D32;font-size:24px;border-radius:2%;"><b>Stayed</b></p>', unsafe_allow_html=True)
    else:
        # st.header('Exited')
        st.markdown(f'<p style="color:#E44D32;font-size:24px;border-radius:2%;"><b>Exited</b></p>', unsafe_allow_html=True)
        
# plot_data = st.button('Plot my Data')

# dataset = pd.read_csv('./Preprocessed_churn_dataset.csv')
# X['exited'] = 2


# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import plotly.express as px

# tsne = PCA(n_components  = 2)

# if plot_data:
#     dd = pd.concat([dataset, X], ignore_index = True, axis = 0)  
#     dataX = dataset.drop('exited', axis = 1)
#     datay = dataset.exited
#     # reducing Dimension
#     reduced_x = tsne.fit_transform(dataX)
#     df = pd.DataFrame(reduced_x, columns=['lat', 'lon'])
#     df['exited'] = datay
#     if 2 in df.exited:
#         st.write('yes')

#     fig = px.scatter(data_frame = df[-30:], x = 'lat', y = 'lon',  color = 'exited')
#     # plt.scatter(x = reduced_x[:,0], y = reduced_x[:,1])
#     st.plotly_chart(fig)


# features = '''
# **Feature Space** \n
# Credit score : Credit score of the customer \n
# Geography : The country from which the customer belongs\n
# Gender : Male or Female \n
# Age : Age of the customer \n
# Tenure : Number of years for which the customer has been with the bank \n
# Balance : Bank balance of the customer \n
# Number of Product : Number of bank products the customer is utilising \n
# Has Credit Card :  customer holds a credit card with the bank or not \n
# Is Active Member :  customer is an active member with the bank or not \n
# Estimated Salary : Estimated salary of the customer in Dollars 
# '''
# st.write(features)
