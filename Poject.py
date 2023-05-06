#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import datetime
import requests
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import glob


# In[2]:


df = pd.read_csv('sensor.csv')
df.info()


# In[3]:


del df['Unnamed: 0']


# In[4]:


df.info()


# In[5]:


df.nunique()


# In[6]:


df['machine_status'].value_counts(dropna=False).head()


# In[7]:


df['sensor_15'].value_counts(dropna=False).head()


# In[8]:


df.describe().T


# In[9]:


df = df.drop_duplicates()


# In[10]:


del df['sensor_15']


# In[11]:


df.shape


# In[12]:


def calc_percent_NAs(df):
    nans = pd.DataFrame(df.isnull().sum().sort_values(ascending=False)/len(df), columns=['percent']) 
    idx = nans['percent'] > 0
    return nans[idx]


# In[ ]:





# In[13]:


calc_percent_NAs(df).head(10)


# In[14]:


df['sensor_50'].fillna((df['sensor_50'].mean()), inplace=True)
df['sensor_51'].fillna((df['sensor_51'].mean()), inplace=True)
df['sensor_00'].fillna((df['sensor_00'].mean()), inplace=True)
df['sensor_08'].fillna((df['sensor_08'].mean()), inplace=True)
df['sensor_07'].fillna((df['sensor_07'].mean()), inplace=True)
df['sensor_06'].fillna((df['sensor_06'].mean()), inplace=True)
df['sensor_09'].fillna((df['sensor_09'].mean()), inplace=True)


# In[15]:


calc_percent_NAs(df).head(10)


# In[16]:


df_clean = df.dropna()


# In[17]:


import warnings
warnings.filterwarnings("ignore")
df_clean['date'] = pd.to_datetime(df_clean['timestamp'])
del df_clean['timestamp']


# In[18]:


df_clean = df_clean.set_index('date')
df_clean.head()


# In[19]:


df_clean.info()


# In[20]:


df_clean.to_csv('cleaned-sensor.csv')


# In[22]:


df=pd.read_csv('cleaned-sensor.csv')


# In[23]:


df.head()


# In[25]:


df.shape


# In[26]:


print(type(df),df['sensor_01'])


# In[27]:


df['date'] = pd.to_datetime(df['date'])


# In[28]:


df.set_index('date', inplace=True)


# In[34]:


import warnings
# Extract the readings from BROKEN state and resample by daily average
broken = df[df['machine_status']=='BROKEN']
# Extract the names of the numerical columns
df2 = df.drop(['machine_status'], axis=1)
names=df2.columns
# Plot time series for each sensor with BROKEN state marked with X in red color
for name in names:
    sns.set_context('talk')
    _ = plt.figure(figsize=(18,3))
    _ = plt.plot(broken[name], linestyle='none', marker='X', color='red', markersize=12)
    _ = plt.plot(df[name], color='green')
    _ = plt.title(name)
    plt.show()


# In[35]:


rollmean = df.resample(rule='D').mean()
rollstd = df.resample(rule='D').std()


# In[36]:


for name in names:
    _ = plt.figure(figsize=(18,3))
    _ = plt.plot(df[name], color='green', label='Original')
    _ = plt.plot(rollmean[name], color='red', label='Rolling Mean')
    _ = plt.plot(rollstd[name], color='black', label='Rolling Std' )
    _ = plt.legend(loc='best')
    _ = plt.title(name)
    plt.show()


# In[49]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import datetime
# scaler = StandardScaler()
# pca = PCA(n_components=2)
date_string = '2018-04-01 00:00:00'

# Convert string to datetime object
date_time_obj = datetime.datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')

# Convert datetime object to float
date_float = float(date_time_obj.date())
data = pd.read_csv('cleaned-sensor.csv')
data_pca = pca.fit_transform(scaler.fit_transform(data))
principalDf = pd.DataFrame(data=data_pca, columns=['pc1', 'pc2'], index=data.index)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(principalDf.values)


# Set the window size and step
window_size = 10
step = 5

# Split the data into sequences with the window size and step
sequences = []
for i in range(window_size, len(scaled_data), step):
    sequences.append(scaled_data[i-window_size:i])

# Convert the list of sequences to numpy array
sequences = np.array(sequences)

# Split the sequences into train and test sets
train_size = int(len(sequences)*0.8)
train_data = sequences[:train_size]
test_data = sequences[train_size:]

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=64, input_shape=(train_data.shape[1], train_data.shape[2]), return_sequences=True))
model.add(LSTM(units=32, return_sequences=False))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(train_data, train_data, epochs=50, batch_size=64)

# Make predictions on the test data
predictions = model.predict(test_data)

# Calculate the reconstruction error
mse = np.mean(np.power(test_data - predictions, 2), axis=1)

# Calculate the threshold for anomaly detection
threshold = np.percentile(mse, 100*(1-outliers_fraction))

# Mark the data points that are anomalies
principalDf['anomaly'] = (mse >= threshold).astype(int)

# Plot the anomaly detection results
_ = plt.figure(figsize=(15, 5))
_ = plt.plot(principalDf.index, principalDf['anomaly'], color='red', label='Anomaly')
_ = plt.plot(principalDf.index, principalDf['pc1'], label='pc1')
_ = plt.xlabel('Time')
_ = plt.ylabel('pc1')
_ = plt.legend()
_ = plt.title('Anomaly Detection Results using LSTM')
plt.show()


# In[39]:




# In[47]:


df.info


# In[48]:


import datetime

# Example date/time string
date_string = '2018-04-01 00:00:00'

# Convert string to datetime object
date_time_obj = datetime.datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')

# Convert datetime object to float
date_float = float(date_time_obj.timestamp())

print(date_float)


# In[50]:


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])


# In[ ]:


scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Apply Factor Analysis with 2 components
fa = FactorAnalysis(n_components=2, random_state=42)
fa_components = fa.fit_transform(x_scaled)
fa_df = pd.DataFrame(data=fa_components, columns=['fa1', 'fa2'])

# Merge the FA components with the machine_status column
merged_df = pd.concat([fa_df, df['machine_status']], axis=1)

# Visualize the FA results
import seaborn as sns
sns.scatterplot(data=merged_df, x='fa1', y='fa2', hue='machine_status')
plt.title('Factor Analysis with 2 components')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




