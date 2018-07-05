
# coding: utf-8

# In[ ]:


# Machine Learning Assignment 1
# Name  (Roll No.)
# HOUSING PRICES


# In[1]:


# Importing Packages
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import recall_score


# In[2]:


# Reading the Data Set
df = pd.read_csv("Housing.csv")


# In[3]:


#Loading Attributes
X = df[['lotsize','bedrooms','bathrms','stories','garagepl']]
y = df['price']


# In[4]:


# Naive Bayes


# In[5]:


gnb = GaussianNB()
Y_pred = gnb.fit(X, y)
prediction = Y_pred.predict(X)
print('Total Data = '+str(df['price'].count()))
print('Wrongly Predicted data = %d'%((y != prediction).sum()))
print('Accuracy = ' + str(accuracy_score(y,prediction)*100))


# In[6]:


# KNN (K Nearest Neighbour)


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[8]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
print('KNN Score : '+str(knn.score(X_test,y_test)*100))


# In[9]:


# SVM


# In[ ]:


model = svm.SVC(kernel='linear',gamma=1)
model.fit(X_train,y_train)
y_predicted = model.predict(X_test)
print('SVM Score : '+str(model.score(X,y)*100))
# print('Recall Score ='+str(recall_score(y_true=y_test,y_pred=y_predicted,average='macro')))

