
# coding: utf-8

# In[1]:

# dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# reading and analysing data
train_file_data = pd.read_csv('train.csv',parse_dates=['Date'])
train_file_data.head()


# In[2]:

train_file_data.describe()


# In[3]:

train_file_data.dtypes


# In[4]:

train_file_data.IsHoliday = train_file_data.IsHoliday.astype(np.int32)
train_file_data.IsHoliday.unique()
train_file_data.head()


# In[5]:

train_file_data['Week'] = train_file_data.Date.dt.week
train_file_data.head()


# In[6]:

copy = train_file_data.copy()
copy.head()
copy = copy.set_index(['Store','Date'])
copy.loc[1,'2014-02-05']


# In[7]:

train_file_data[(train_file_data['Store'] == 7) & (train_file_data['Date'] == '2014-02-05')]['Dept'].max()


# In[8]:


# Number of department for each store are not equal
# Number of departments for a store are not equal for different dates

#Plot department wise data for all stores
# get_ipython().magic('matplotlib inline')
plt.figure(figsize=(20,10))
for i in train_file_data['Store'].unique()[:10]:   #For 10 stores
    plt.subplot(2, 1, 1)
    plt.title('Sales Vs Department')
    plt.scatter(train_file_data[(train_file_data['Store'] == i) & (train_file_data['Date'] == '2014-02-05')]['Dept'],train_file_data[(train_file_data['Store'] == i) & (train_file_data['Date'] == '2014-02-05')]['Sales'])
    plt.ylabel('Sales')
    plt.xlabel('Department')
    plt.subplot(2, 1, 2)
    plt.plot(train_file_data[(train_file_data['Store'] == i) & (train_file_data['Date'] == '2014-02-05')]['Dept'],train_file_data[(train_file_data['Store'] == i) & (train_file_data['Date'] == '2014-02-05')]['Sales'])
    plt.ylabel('Sales')
    plt.xlabel('Department')
#     break  # comment/uncomment this to plot for all/first stores


    
# Inference: the The sales is moderate for some departments Initially  i.e till department 18 then sales decreases
# There is a occasional rise in sales for each store around department 38-40 then around 72-73 then from 90-100


# In[9]:

# plot the sales of particular store departments of a year

plt.figure(figsize=(20,5))

for i in train_file_data['Dept'].unique()[:2]: # for 2 departments
    store_no = 3
    x = train_file_data[(train_file_data['Store'] == store_no) & (train_file_data['Dept'] == i)]['Week']
    y = train_file_data[(train_file_data['Store'] == store_no) & (train_file_data['Dept'] == i)]['Sales']
    plt.scatter(x,y)
    plt.xlabel('Week Number')
    plt.ylabel('Sales')
#     break #  Comment/Uncomment for one/all plot

# Conclusion : In this plot department 2 shows greater consistenstency in sales so is more likely to be a general
#              store as compared to department 1 which has greater sales during winters


# In[10]:

# Plot sales on a day of holiday

print(train_file_data[(train_file_data['IsHoliday'] == 1) & (train_file_data['Store'] == 1)]['Sales'].mean())
print(train_file_data[(train_file_data['IsHoliday'] == 0) & (train_file_data['Store'] == 1)]['Sales'].mean())

# Conclusion: Sales is greater on holidays which is understandable as people are more free for shopping


# In[11]:

# loading store features data.
features = pd.read_csv('features.csv',parse_dates=['Date'])
features.head()


# In[12]:

# Checking datatypes of columns
features.dtypes


# In[13]:

# Getting some analytics data
features.describe()


# In[14]:

# Checking null or NaN values if any
features.isnull().any()


# In[15]:

# dropping Unemployment and Fuel_price data as they do not affect sales much
del features['Unemployment']
del features['Fuel_Price']
features.head()


# In[16]:

features['Week'] = features.Date.dt.week
copy_features = features.copy() # Will combine data using this copy_features to avoid any damage to main dataframe 
copy_features = copy_features.set_index(['Store','Date'])
copy_features.head()


# In[17]:

copy_features.loc[1,'2014-02-05T00:00:00.000000000']


# In[18]:

copy['Temperature'] = 'nan'
copy['CPI'] = 'nan'
copy.loc[(1,'2014-02-05')]['Temperature']


# In[19]:

import os       # To check if merged datframe file exists or not
# Merging data frames
if not os.path.isfile('merged_data.csv'):
    sorted_dates = np.sort(train_file_data.Date.unique())
    for i in range(1,46):
        for j in sorted_dates:
    #         print(i,j)
            copy.loc[(i,j),'Temperature'] = copy_features.loc[i,str(j)]['Temperature']
            copy.loc[(i,j),'CPI'] = copy_features.loc[i,str(j)]['CPI']
    #         break
    #     break    
    copy
else:
    main = pd.read_csv('merged_data.csv')
    copy = main
    copy


# In[20]:

# Saving DataFrame
if not os.path.isfile('merged_data.csv'):
    copy.to_csv('merged_data.csv')


# In[21]:

# Extracting sales and dropping sales from the dataframe
sales = copy['Sales']
sales.head()


# In[22]:

del copy['Sales']
del copy['Date']
copy.head()


# In[23]:

#breaking into training and testing data and converting to numpy array to use in sklearn models
train_len = int(len(copy)*.75)
train_features = copy[:train_len].as_matrix()
test_features = copy[train_len:len(copy)].as_matrix()
train_res = sales[:train_len].as_matrix()
test_res = sales[train_len:len(copy)].as_matrix()
print(len(train_features),len(test_features),len(train_res),len(test_res))
print(sales[0])
train_features


# # In[24]:

# #Importing Machine Learning Models
# from sklearn import linear_model
# from sklearn.svm import SVR

# # Linear Regression Model
# model1 = linear_model.LinearRegression()
# model1.fit(train_features,train_res)
# model1.score(train_features,train_res)


# # In[25]:

# # # Support Vector Regression Model
# # model2 = SVR()
# # model2.fit(train_features,train_res)
# # model2.score(test_features,test_res)


# # In[26]:

# #Ridge Regression
# model3 = linear_model.Ridge(alpha = 0.5)
# model3.fit(train_features,train_res)
# model3.score(test_features,test_res)


# # In[27]:

# #RidgeCV
# model4 = linear_model.RidgeCV()
# model4.fit(train_features,train_res)
# print(model4.alpha_)
# model4.score(test_features,test_res)


# # In[28]:

# #Bayesian Regression
# model5 = linear_model.BayesianRidge()
# model5.fit(train_features,train_res)
# model5.score(test_features,test_res)
# model5.predict([1,1,0,6,5.73,211.09])


# In[29]:

# model5.coef_


# In[ ]:

#_______________________________MODEL TRAINING_______________________________________________________

from keras.models import Sequential
from keras.layers import Dense
import math

model = Sequential()
model.add(Dense(8,input_dim=6,activation='relu'))
model.add(Dense(8,input_dim=8,activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(train_features,train_res,epochs=50,batch_size=50,verbose=1)

# Estimate model performance
trainScore = model.evaluate(train_features,train_res, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(test_features, test_res, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

#Save model to json file
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

#________________________________________________________________________________________________________



