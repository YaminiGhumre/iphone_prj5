#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib as plt
import numpy as np
from sklearn import preprocessing

######## Take you mpg data and load it over here - 
data = pd.read_csv('E:/Python docs/iphone_purchase_records.csv')
print(data.head(5))


# In[8]:


##### KNN Regression
y = data['Purchase Iphone']
X = data[['Age','Salary']]


# In[9]:


y.shape


# In[10]:


X.shape


# In[41]:


from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=123)


# In[42]:


y_train.shape


# In[43]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score


# In[44]:


model = KNeighborsRegressor(n_neighbors=2) ###### default
print("The model is loaded")


# In[45]:


###### Training the Model
model_fitting = model.fit(X_train, y_train)
print('Model Training is completed')


# In[46]:


model_fitting.score(X_train, y_train)


# In[47]:


round(model_fitting.score(X_train, y_train),2)


# In[48]:


###### Prediction  - Testing Data
pred = model_fitting.predict(X_test)
results = r2_score(y_test,pred)
print(results)  #### 0 to 1


# In[49]:


data.shape


# In[50]:


##### Error Method -  k =  (1,21) It will give me the error present in those k
error = []
k = []
for i in range(1,10,2): #### K values  #### stepover
    model = KNeighborsRegressor(n_neighbors=i)
    model_fit = model.fit(X_train,y_train)
    err = 1 - round(model_fit.score(X_train,y_train),2)
    error.append(err)
    k.append(i)


# In[51]:


pd.DataFrame({'K':k, 'error':error})


# In[52]:


##### How to Save the Model ---
import joblib


# In[53]:


joblib.dump(model,'KNN_Reg_model.sav')


# In[54]:


#!pip install joblib  ##### To install Joblib


# In[55]:


X_test


# In[56]:


X_test['Actual'] = y_test
X_test['Pred'] = pred


# In[57]:


X_test


# In[58]:


import joblib


# In[59]:


joblib.dump(model,'iphone_prj5.sav')


# In[60]:


X_test


# In[ ]:





# In[ ]:




