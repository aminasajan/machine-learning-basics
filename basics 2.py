#!/usr/bin/env python
# coding: utf-8

# In[1]:


#https://raw.githubusercontent.com/ameenmanna8824/DATASETS/main/areavsprices.csv
#Area-Sfft
#prices-rupees


# In[3]:


#1.take the data an create a dataframe
import pandas as pd
df=pd.read_csv('https://raw.githubusercontent.com/ameenmanna8824/DATASETS/main/areavsprices.csv')
df


# In[4]:


#2.pre processing (EDA)-HERE NOT REQUIRED


# In[7]:


#3.data visualization
import matplotlib.pyplot as plt
plt.scatter(df['Area'],df['Prices'])
plt.title("area vs prices")
plt.xlabel('area')
plt.ylabel('prices')


# In[8]:


#4.divie the data into output an input

#input(x)- is always 2d array-area
#input(y)- is always 1d array-prices


# In[12]:


x=df.iloc[:,0:1]
x


# In[13]:


x=x=df.iloc[:,0:1].values
x


# In[15]:


y=df.iloc[:,1].values
y


# In[16]:


#5.train an test variables
#6.normalization-only for multivariate-not required


# In[19]:


#7. run a classifier,regressor,clusterer
from sklearn.linear_model import LinearRegression
model=LinearRegression()


# In[20]:


#8. fit a model(mapping/plotting ip with op)
model.fit(x,y)


# In[21]:


#9.predict the op
y_pred=model.predict(x)
y_pred


# In[22]:


y


# In[23]:


#10.evaluation
model.predict([[2000]])


# In[24]:


#cross verification technique
#y=mx+c


# In[25]:


#slope m
m=model.coef_
m


# In[26]:


c=model.intercept_
c


# In[27]:


#substitute
m*2000+c


# In[33]:


#visualization
plt.scatter(x,y,c='orange')
plt.plot(x,y_pred)


# In[ ]:





# In[ ]:




