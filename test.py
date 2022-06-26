
import joblib
# genders = input('Your Gender',['Male','Female'])
# geographys = input( 'Your REGION',['France', 'Spain', 'Germany'])

gender = 'Female'
geography = 'Germany'

import numpy as np

x = np.array([geography, gender]) 

print(x)

# load the saved components # scaler and One hot encoder
ohe = joblib.load('./Saved_components/OneHotEncoder.joblib')
scaler = joblib.load('./Saved_components/Scaler.joblib')
encoded = ohe.transform([x])
print(*(np.ravel(encoded)))

