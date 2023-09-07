#!/usr/bin/env python
# coding: utf-8

# ## Machine learning project 
# ## By Ravendrika Mahapatra
# ## Wine Prediction Model 

# In[50]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load the data

# In[4]:


df = pd.read_csv("winequality-red.csv")
df


# # Data Preprocessing and Data Visualization

# In[5]:


df.shape


# In[6]:


df.head()


# In[7]:


df.describe()


# In[8]:


df.columns


# In[9]:


df.isnull().sum()


# In[12]:


sns.catplot(x = 'quality',data=df,kind='count')


# In[20]:


plot= plt.figure(figsize=(5,5))
sns.barplot(x ='quality',y ='volatile acidity',data = df)


# In[19]:


plot = plt.figure(figsize=(5,5))
sns.barplot(x ='quality',y ='citric acid',data=df)


# In[54]:


plot = plt.figure(figsize=(5,5))
sns.barplot(x ='quality',y='residual sugar',data = df)


# In[56]:


plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality',y = 'chlorides',data=df)


# In[57]:


plot = plt.figure(figsize =(5,5))
sns.barplot(x='quality', y='pH',data=df)


# In[58]:


plot = plt.figure(figsize =(5,5))
sns.barplot(x='quality', y='total sulfur dioxide',data=df)


# In[59]:


plot = plt.figure(figsize =(5,5))
sns.barplot(x='quality', y='sulphates',data=df)


# # Data Correlation
# ## Positive Correlation
# ## Negative Correlation

# In[21]:


correlation= df.corr()


# In[28]:


fig = plt.subplots(figsize=(10,10))
sns.heatmap(correlation,vmax=1,square=True,annot=True,cmap='Reds')


# In[97]:


# Create a box plot
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, orient='v')
plt.title("Box Plot")
plt.show()


# In[29]:


X= df.drop('quality',axis=1)


# In[30]:


print(X)


# ## Label Binarization

# In[33]:


Y = df['quality'].apply(lambda y_value: 1 if y_value>=7 else 0 )


# In[34]:


print(Y)


# ## Train and Test Split

# In[72]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size =0.2,random_state = 3)


# In[73]:


print(Y.shape,Y_train.shape,Y_test.shape)


# ## Model Training
# # Random Forest Classifier

# In[74]:


model = RandomForestClassifier()
 


# In[75]:


model.fit(X_train,Y_train)


# ## Model Prediction 
# ## Accuracy Score

# In[76]:



X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)


# In[77]:


print(test_data_accuracy)


# ## Binding a Predictive System

# In[91]:


input_data=(7.8,0.58,0.02,2.0,0.073,9.0,18.0,0.9968,3.36,0.57,9.5)
# changing the input data of numpy array
input_data_as_numpyarray = np.asarray(input_data)
input_data_reshape = input_data_as_numpyarray.reshape(1,-1)
prediction = model.predict(input_data_reshape)
print(prediction)


# In[ ]:





# In[ ]:




