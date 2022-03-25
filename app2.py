import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
#st.title("Salary Prediction")
st.write("""
# Stock Price Prediction App
This app predicts the **Stock Price with input of Opening price of NIFTY 50** 
""")

st.sidebar.header('User Input Parameters')
def user_input_features():
    ClosePrice = st.sidebar.slider('Open', 0, 15000, 12200)
    data = {'Close Price': ClosePrice}
    features = pd.DataFrame(data, index=[0])
    return features
df = user_input_features()

st.subheader('User Input parameters(Input Opening Price Of NSE)')
st.write(df)

dataset = pd.read_csv("Copydata6.csv")
X = dataset[["Open"]]
y = dataset[["Close Price"]]

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

y_pred = regressor.predict(X)

from sklearn.metrics import r2_score
r2 = r2_score(y , y_pred)
print(r2)

Prediction = regressor.predict(df)
st.subheader('Prediction of Stock Price at the end of the day')
st.write(Prediction)