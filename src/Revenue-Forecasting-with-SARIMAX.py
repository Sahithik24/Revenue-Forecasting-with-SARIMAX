# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# LOADING THE DATA
FILEPATH = ("churn_clean.csv")
df = pd.read_csv(FILEPATH)
df.info()


# In[3]:


# MISSING VALUES
null_values = df.isna().sum()
print(null_values)


# In[4]:


# DUPLICATE VALUES
num_duplicates = df.duplicated().sum()
print(num_duplicates)


# In[5]:


print(df)


# In[6]:


# CONVERTING DAY TO DATE
start_date = pd.to_datetime('2022-12-31')  
df['Date'] = start_date + pd.to_timedelta(df['Day'], unit='D')
print(df)


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


# SETTING DATE AS INDEX
df.set_index('Date', inplace=True)


# In[9]:


# PLOTTING THE LINE GRAPH
plt.plot(df.index, df['Revenue'], marker='o', linestyle='-')
plt.title('Revenue Time Series')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[10]:


from statsmodels.tsa.stattools import adfuller


# In[11]:


# ADF TEST
result = adfuller(df['Revenue'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])


# In[12]:


# FINDING PEAK
peak_index = df['Revenue'].idxmax()

# SPLITTING THE DATA
train = df.loc[:peak_index]   
test = df.loc[peak_index + pd.Timedelta(days=1):] 


# In[13]:


# Save the CLEANED DATA 
train.to_csv('train_data(TASK3).csv')
test.to_csv('test_data(TASK3).csv')


# In[14]:


train['Revenue'].plot()  


# In[15]:


from statsmodels.graphics.tsaplots import plot_acf


# In[16]:


# AUTOCORRELATION FUNCTION
plot_acf(train['Revenue'], lags=40)
plt.title('Autocorrelation Function of Revenue')
plt.grid(True)
plt.show()


# In[17]:


from scipy.signal import periodogram


# In[18]:


#SPECTRAL DENSITY
freqs, psd = periodogram(train['Revenue'])

plt.plot(freqs, psd)
plt.title('Spectral Density of Revenue Time Series')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.grid(True)
plt.show()


# In[19]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[20]:


#DECOMPOSITION 
decomposition = seasonal_decompose(train['Revenue'], model='additive', period=30)
decomposition.plot()
plt.suptitle('Decomposition of Revenue Time Series', fontsize=14)
plt.show()


# In[21]:


#CONFIRMATION OF DECOMPOSITION
plt.plot(decomposition.resid)
plt.title('Residuals from Decomposition')
plt.xlabel('Date')
plt.ylabel('Residual Value')
plt.grid(True)
plt.show()


# In[22]:


#converting into stationary 
# First differencing
train_diff = train['Revenue'].diff().dropna()


# In[23]:


# ADF TEST
result1 = adfuller(train_diff)
print('ADF Statistic:', result1[0])
print('p-value:', result1[1])
print('Critical Values:', result1[4])


# In[24]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[25]:


# ACF plot
plot_acf(train_diff, lags=40)
plt.title('ACF Plot after Differencing')
plt.show()


# In[26]:


# PACF plot
plot_pacf(train_diff, lags=40)
plt.title('PACF Plot after Differencing')
plt.show()


# In[27]:


from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[28]:


# DEFINING SARIMAX
model = SARIMAX(train['Revenue'],
                order=(1, 1, 2),              
                seasonal_order=(1, 1, 1, 30),   
                enforce_stationarity=False,
                enforce_invertibility=False)

# FIT MODEL
model_fit = model.fit()

# SUMMARY
print(model_fit.summary())


# In[32]:


start = len(train)
end = len(train)+len(test)-1
pred = model_fit.predict(start=start, end=end)

#CONFIDENCE INTERVAL
pred_result = model_fit.get_prediction(start=start, end=end)
conf_int = pred_result.conf_int(alpha=0.05) 

#INDEX
pred.index = df.index[start:end+1]
conf_int.index = df.index[start:end+1]

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(test.index, test['Revenue'], label='Actual Revenue', linewidth=2)
plt.plot(pred.index, pred, label='Predicted Revenue', linestyle='--', linewidth=2)

# 95% Confidence Interval
plt.fill_between(conf_int.index,
                 conf_int.iloc[:, 0],
                 conf_int.iloc[:, 1],
                 color='red', alpha=0.1, label='95% CI')

# Formatting
plt.title('Actual vs Predicted Revenue with 95% Confidence Intervals')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[41]:


from sklearn.metrics import mean_absolute_error, mean_squared_error

# PREDICTION ON TRAIN DATA
train_prediction = model_fit.predict(start=0, end=len(train) - 1)

# PREDICTING ON TEST DATA
start_test = len(train)
end_test = len(train) + len(test) - 1
test_prediction = model_fit.predict(start=start_test, end=end_test)

# MAE and RMSE FOR TRAINING SATA SET
train_mae = mean_absolute_error(train['Revenue'], train_prediction)
train_rmse = np.sqrt(mean_squared_error(train['Revenue'], train_prediction))

# MAE and RMSE FOR TEST DATA SET
test_mae = mean_absolute_error(test['Revenue'], test_prediction)
test_rmse = np.sqrt(mean_squared_error(test['Revenue'], test_prediction))

print(f"Training MAE: {train_mae:.2f}")
print(f"Training RMSE: {train_rmse:.2f}")
print(f"Test MAE: {test_mae:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")


# In[37]:


# FORECASTING 90 DAYS
forecast_steps = 90
forecast_result = model_fit.get_forecast(steps=forecast_steps)

forecast_mean = forecast_result.predicted_mean.astype(float)
conf_int = forecast_result.conf_int().astype(float)
conf_int.columns = ['lower', 'upper']
forecast_index = pd.date_range(start=test.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')

# PLOTING FORECAST GRAPH
plt.figure(figsize=(14, 7))
plt.plot(train.index, train['Revenue'], label='Training Data', color='blue')
plt.plot(test.index, test['Revenue'], label='Test Data', color='orange')
plt.plot(forecast_index, forecast_mean, label='Forecast (90 Days)', color='red', linestyle='--')
plt.fill_between(forecast_index,
                 conf_int['lower'], conf_int['upper'],
                 color='red', alpha=0.2, label='Confidence Interval')
plt.title('90 - Day Forecast Plot')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
