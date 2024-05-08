import numpy as np 
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model 
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import streamlit as st 
from sklearn.preprocessing import MinMaxScaler

#Load Model 
model = load_model(r'C:\Users\Altair\PythonCodes\Quantitative_Finance_Projects\LSTM_Stock_Prediction_Web_App\Stock_Predictions_Model.keras')

# Create Inputs 
st.header('Stock Market Predictor')
stock = st.text_input('Enter Stock Symbol', 'TSLA')

#LOad data
num_of_years_back = 10
end_date = datetime.today()
start_date = end_date - timedelta(days=365 * num_of_years_back)
data = yf.download(stock, start_date, end_date)

st.subheader('Stock Data')
st.write(data)

# Split 
data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.8)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.8): len(data)])

# Scale data 
scaler = MinMaxScaler(feature_range = (0,1))
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test] , ignore_index = True)
data_test_scale = scaler.fit_transform(data_test)

# preprocess the data to feed the model

st.subheader('Price vs Moving Average - 50 & 100')

ma_50_dats = data.Close.rolling(50).mean()
ma_100_days = data.Close.rolling(100).mean()
ma_200_days = data.Close.rolling(200).mean()

fig1 = plt.figure(figsize = (8,6))
plt.plot(ma_50_dats, 'r', label = 'MA50')
plt.plot(ma_100_days, 'b', label = 'MA100')
plt.plot(data.Close, 'g')
plt.ylabel('Price')
plt.title('SMA - 50 and 100')
plt.legend()
plt.show()

st.pyplot(fig1)


x = []
y = []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x, y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_
predict = predict * scale 
y = y*scale 

# Plot results 

st.subheader('Predicted vs Real Price')

fig2 = plt.figure(figsize = (8,6))
plt.plot(predict, 'r', label = 'Predicted Price')
plt.plot(y, 'b', label = 'Real Value')
plt.ylabel('Price')
plt.xlabel('Time')
plt.legend()
plt.show()

st.pyplot(fig2)

