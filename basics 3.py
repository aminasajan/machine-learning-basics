#!/usr/bin/env python
# coding: utf-8

# In[1]:


#classification -logistic regression(supervised)
#https://raw.githubusercontent.com/ameenmanna8824/DATASETS/main/Social_Network_Ads.csv


# In[3]:


#1. take data an create a datafrmae
import pandas as pd
df=pd.read_csv("https://raw.githubusercontent.com/ameenmanna8824/DATASETS/main/Social_Network_Ads.csv")
df


# In[4]:


df.shape


# In[5]:


df.size


# In[6]:


df.info()


# In[7]:


#3. no need for classification


# In[10]:


#input-age and est salary
#output- purchased


# In[9]:


#4.divide into ip and op


# In[14]:


x=df.iloc[:,2:4].values
x


# In[18]:


y=df.iloc[:,4].values
y


# In[19]:


df['Purchased'].value_counts()


# In[20]:


#5
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)


# In[23]:


x.shape


# In[24]:


x_train.shape


# In[25]:


x_test.shape


# In[26]:


y.shape


# In[27]:


y_train.shape


# In[28]:


y_test.shape


# In[29]:


#6. normalization
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)


# In[31]:


#7. apply classifier,regressor or clusterer
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[32]:


#8. fit the moel
model.fit(x_train,y_train)


# In[33]:


#9. preict the output
y_pred=model.predict(x_test)
y_pred


# In[34]:


y_test


# In[38]:


#10. accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)*100


# In[39]:


#iniviual preiction
a=scaler.transform([[25,50000]])
a


# In[40]:


model.predict(a)


# In[41]:


b=scaler.transform([[50,35000]])
model.predict(b)


# In[42]:


#classification model 2
#1. take data an create a dataframe
import pandas as pd
df=pd.read_csv("https://raw.githubusercontent.com/ameenmanna8824/DATASETS/main/IRIS.csv")
df


# In[43]:


df.shape


# In[44]:


df.size


# In[45]:


df.info()


# In[50]:


#4.divide into input and output
df["species"].value_counts()


# In[51]:


#input-sepal length, sepal width, petal length, petal width
#output-species


# In[52]:


x=df.iloc[:,0:4]
x


# In[53]:


y=df.iloc[:,4]
y


# In[54]:


#5. test an train variable


# In[56]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)


# In[62]:


x.shape


# In[60]:


x_train.shape


# In[61]:


x_test.shape


# In[66]:


y.shape


# In[67]:


y_train.shape


# In[68]:


y_test.shape


# In[70]:


#7. run a classifier/ clusterer/ regressor
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[71]:


#8. fit the model
model.fit(x_train,y_train)


# In[72]:


#9. preict the output
y_pred=model.predict(x_test)
y_pred


# In[73]:


from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)*100


# In[75]:


#iniviual preiction
model.predict([[5.1, 3.5, 1.4 ,0.2]])


# In[78]:


model.predict([[2.1, 4.5, 6.4 ,1.2]])


# In[6]:


import cv2
img=cv2.imread('pic2.jpg')
cv2.imshow("out",img)
cv2.waitKey(3000)
cv2.destroyAllWindows()


# In[1]:


import cv2
print(cv2.__version__)


# In[11]:


import cv2
img=cv2.imread('download.jpg')
cv2.imshow("out",img)
cv2.waitKey(3000)
cv2.destroyAllWindows()


# In[31]:


import cv2
import numpy as np
img=cv2.imread(r'C:\Users\DELL\Documents\personal certificates\amina_photo.jpg')


# In[32]:


cv2.imshow('out',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[33]:


import cv2
import numpy as np
img=cv2.imread(r'C:\Users\DELL\Documents\personal certificates\amina_signature.jpg')
cv2.imshow('out',img)
cv2.waitKey(3000)
cv2.destroyAllWindows()


# In[34]:


import cv2
img=cv2.imread(r'C:\Users\DELL\Documents\personal certificates\amina_signature.jpg')
print(img.shape)


# In[35]:


#357 height, 355 with 3 epth


# In[38]:


#4. grayscale image
import cv2
img=cv2.imread(r'C:\Users\DELL\Documents\personal certificates\amina_photo.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('orginal',img)
cv2.imshow('gray scale',gray)
cv2.waitKey(6000)
cv2.destroyAllWindows()


# In[39]:


import cv2
img=cv2.imread(r'C:\Users\DELL\Documents\personal certificates\amina_photo.jpg',0)

cv2.imshow('gray scale',img)
cv2.waitKey(6000)
cv2.destroyAllWindows()


# In[2]:


#5.binary image conversion(high contrast)
import cv2
gray=cv2.imread(r'C:\Users\DELL\Documents\personal certificates\amina_photo.jpg',0)
cv2.imshow('out',gray)
ret,binary=cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
cv2.imshow('BINARY',binary)
cv2.waitKey(8000)
cv2.destroyAllWindows()


# In[3]:


#6. solid background(w&b bg)
import cv2
import numpy as np
img=np.ones((500,500,3))
cv2.imshow('white background',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[4]:


#6. solid background(w&b bg)
import cv2
import numpy as np
img=np.zeros((500,500,3))
cv2.imshow('black background',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[9]:


#7. soli colors(red,green an blue)
#red color
import cv2
import numpy as np
img=np.zeros((150,150,3))
cv2.imshow('org',img)

img[:]=0,0,255 #b,g,r
cv2.imshow('red',img)
cv2.waitKey(3000)
cv2.destroyAllWindows()


# In[10]:


#7. soli colors(red,green an blue)
#green color
import cv2
import numpy as np
img=np.zeros((150,150,3))
cv2.imshow('org',img)

img[:]=0,255,0 #b,g,r
cv2.imshow('green',img)
cv2.waitKey(3000)
cv2.destroyAllWindows()


# In[12]:


#7. soli colors(red,green an blue)
#yellow color
import cv2
import numpy as np
img=np.zeros((150,150,3))
cv2.imshow('org',img)

img[:]=0,2,2 #b,g,r
cv2.imshow('yellow',img)
cv2.waitKey(3000)
cv2.destroyAllWindows()


# In[16]:


#checker board
import cv2
import numpy as np
img=np.zeros((200,200,3))
img[0:100,0:100]=255,255,255 #white
img[100:200,100:200]=255,255,255
cv2.imshow('checker board',img)
cv2.waitKey(3000)
cv2.destroyAllWindows()


# In[20]:


import cv2
import numpy as np
img=cv2.imread(r'C:\Users\DELL\Documents\personal certificates\amina_photo.jpg')
cv2.imshow('org',img)
cv2.waitKey(3000)
img1=cv2.resize(img,None,fx=0.75,fy=0.75)
cv2.imshow('downscaled image',img1)
img2=cv2.resize(img,None,fx=1.5,fy=1.5)
cv2.imshow('upscaled image',img2)
cv2.waitKey(3000)
cv2.destroyAllWindows()


# In[ ]:


import cv2
import numpy as np
img=cv2.imread(r'C:\Users\DELL\Documents\personal certificates\amina_photo.jpg')
cv2.imshow('org',img)
cv2.waitKey(3000)
img1=cv2.resize(img,None,fx=0.75,fy=0.75)
cv2.imshow('downscaled image',img1)
cv2.waitKey(3000)
img2=cv2.resize(img,None,fx=1.5,fy=1.5)
cv2.imshow('upscaled image',img2)
cv2.waitKey(3000)
#custom dimensions
img3=cv2.resize(img,(1000,4000))
cv2.imshow('customized image',img3)
cv2.waitKey(3000)
cv2.destroyAllWindows()


# In[ ]:


import cv2
import numpy as np
img=np.zeros((500,500,3))
cv2.rectangle(img,(200,200),(400,400),(0,0,255),5)
cv2.imshow('rectangle',img)
cv2.waitKey(3000)
cv2.destroyAllWindows()


# In[ ]:


#.FACE DETECTION IN AN IMAGE
import cv2


face_cascade = cv2.CascadeClassifier(r'C:\Users\DELL\Documents\SECOND\haarcascade_frontalface_default.xml')#importing haarcascade mode
img = cv2.imread(r'C:\Users\DELL\Documents\SECOND\abc.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray,1.1,9)
#1.1 - ScalerFactor,9 is minNeighbors  -- Tuning Parameters


for x,y,w,h in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)
        #cv2.rectangle(src,start pt,end pt,color,thickness)


cv2.imshow('Face Detection',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




