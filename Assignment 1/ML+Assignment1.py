
# coding: utf-8

# In[ ]:


# Machine Learning Assignment 1
# Name (Roll No.)
# HOUSING PRICES


# In[1]:


# Importing Packages
import matplotlib
import matplotlib.pyplot as plt


# In[2]:


import numpy as np
from sklearn import datasets, linear_model
import pandas as pd


# In[3]:


# Reading the Data Set
df = pd.read_csv("Housing.csv")


# In[4]:


# Loading Columns/Attributes

Y = df['price']
# X = df[['lotsize','bedrooms','bathrms','stories','garagepl']]
X = df['lotsize']
X=X.values.reshape(len(X),1)
Y=Y.values.reshape(len(Y),1)


# In[5]:


# plotting Histograms

# plt.figure(figsize(20,20))
plt.hist(X, 50, normed=1, facecolor='g', alpha=0.75)
plt.grid(True)
plt.show()


# In[6]:


plt.hist(Y, 50, normed=1, facecolor='r', alpha=0.75)
plt.grid(True)
plt.show()


# In[7]:


# Plotting the Scatter plot

plt.scatter(X, Y,  color='blue', alpha=0.4)
plt.xticks(())
plt.yticks(())
 
plt.show()


# In[8]:


# Splitting the data into test and train data

X_train = X[:-250]
X_test = X[-250:]

Y_train = Y[:-250]
Y_test = Y[-250:]


# In[9]:


# Plotting the Scatter plot for test data

plt.scatter(X_test, Y_test,  color='red', alpha=0.5)
plt.title('Test Data')
plt.xlabel('Area')
plt.ylabel('Price')
plt.xticks(())
plt.yticks(())
plt.show()


# In[10]:


plt.scatter(X_test, Y_test,  color='red', alpha=0.4) 

# Creating the linear regression object
obj = linear_model.LinearRegression()

# Train the model using the training sets 
obj.fit(X_train, Y_train)
pred = obj.predict(X_test)

# plotting the Outputs
plt.plot(X_test, pred, color='blue')


# In[11]:


plt.title('Test Data')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()


# In[12]:


# checking the score
obj.score(X_test,Y_test)

