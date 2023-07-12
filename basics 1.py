#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
a=[1,2,3,4]
b=[5,6,7,8]
plt.plot(a,b,'red')


# In[20]:


import matplotlib.pyplot as plt
a=[1,2,3,4]
b=[5,6,7,8]
plt.plot(a,b,c='red',marker="*")
plt.title('line graph')
plt.xlabel("x-axis")
plt.ylabel("Y-axis")
plt.xticks(range(0,20,2))
#plt.yticks(range(0,20,2))


# In[13]:


import numpy as np
x=np.array([1,2,3,4])
y=np.array([5,6,7,8])
plt.plot(x,y,c="orangered",marker="*")
plt.title("line graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")


# In[19]:


a=[1,2,3,4]
b=[5,6,7,8]
plt.scatter(a,b,c=["orange",'green','gold',"red"])
plt.title("scatter plot")
plt.xlabel("X axis")
plt.ylabel("Y axis")


# In[26]:


names=["a",'b','c','e']
weight=[45,34,67,47]
plt.bar(names,weight,color=["orange",'green','gold',"lime"])
plt.title("bar graph")
plt.xlabel('names')
plt.ylabel('weight')
#barh


# In[29]:


mydataset={'cars':['BMW','Volvo','Ford'],'passings':[3,7,2]}
mydataset


# In[30]:


type(mydataset)


# In[34]:


import pandas as pd
df= pd.DataFrame(mydataset)
df


# In[35]:


type(df)


# In[37]:


a=[1,7,3,5]
myvar1=pd.Series(a)
myvar1


# In[1]:


#https://storage.googleapis.com/kagglesdsdata/datasets/9590/13660/fruit_data_with_colors.txt?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230708%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230708T125111Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=ad8c1923fbb9ddf6ade3b1d84737a86be70b946ac3c9a82ac7c8746a9970ff9a3617195fa9579b8f4b7eaa85c69ff64d7071037a1a6824ab9c75cd48fa331298eb62d7f1d3b82c382e13adc4366b9f30487415a2782cf463d60dcf00c7c145664b48b4611612072359c48b10e6a8600f38f4bb9fd04e5c52d32cc5267985faae349e5751ff8912b9989bf5540ee2fb976d00d1669213b8904e14b4867906253c8f9916b34f7e12ad60dcb615b53eeba144963ec160eb266f3edb5d2f16fd7e35e59f5661175e47f83e935b4dad8f3e5d43904879ca892658b9dee4b9d1606e02800efbdb8639ab30f2669c39984a8058f386ae9105622f39bd65c6c21500e26f


# In[2]:


https://storage.googleapis.com/kagglesdsdata/datasets/9590/13660/fruit_data_with_colors.txt?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230708%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230708T125111Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=ad8c1923fbb9ddf6ade3b1d84737a86be70b946ac3c9a82ac7c8746a9970ff9a3617195fa9579b8f4b7eaa85c69ff64d7071037a1a6824ab9c75cd48fa331298eb62d7f1d3b82c382e13adc4366b9f30487415a2782cf463d60dcf00c7c145664b48b4611612072359c48b10e6a8600f38f4bb9fd04e5c52d32cc5267985faae349e5751ff8912b9989bf5540ee2fb976d00d1669213b8904e14b4867906253c8f9916b34f7e12ad60dcb615b53eeba144963ec160eb266f3edb5d2f16fd7e35e59f5661175e47f83e935b4dad8f3e5d43904879ca892658b9dee4b9d1606e02800efbdb8639ab30f2669c39984a8058f386ae9105622f39bd65c6c21500e26f


# In[4]:


import pandas as pd
df=pd.read_csv(' https://storage.googleapis.com/kagglesdsdata/datasets/9590/13660/fruit_data_with_colors.txt?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230709%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230709T050050Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=353f9aa7192976f35934e883ce26371e28848bb022b77f67fc1bcf7229b5bbd8c409ba71c0add8bb2b7645c1799514471002cb869d005f90e8f46993512b5df8f01e5d7a6839b5fde3abf9a9ce1131abeeaddc16afc6c3722abe7e751e5ad6a4edddb68697d3c9c7e69d7002536cef9c8e99d0be16f043445902153b97ee2650f34f2cace35d21f34269b13a39ad74d89e45c04a016760619ffcf85f3d8ca107d2a5c1e634777133e307f6f22138fd3934e98f3f1d51d9c44784784791e676c2a207e1b581fbc3cc0f1a63c7f0d477b2c0c517f97c26a635b60a3639bfc1529ce14443312a870be9f37b505e8f16da1c7dc5da60706c97488809e848c0163197 ')
df
               


# In[1]:


storage.googleapis.com_kagglesdsdata_datasets_9590_13660_fruit_data_with_colors.txt_X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com2F202307092Fstorage%2Fgoog4_request&X-Goog-Date=
    


# In[ ]:





# In[44]:


import os

# Specify the relative or absolute path of the text file
file_path = 'https://storage.googleapis.com/kagglesdsdata/datasets/9590/13660/fruit_data_with_colors.txt?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230709%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230709T061546Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=1292994608c367ee34eb345ff45469234c7092efce51aa618d6975d1290411683fafa7637add754de7b035084df99004275627f6a9f25a3fbdec876cd3622b74786013ae732334a91a442046665f5c59ce5072574fc1dc9d896f8f0d4be2e53c0de945c18c1bd38719513a7464f341f82f1c8712ea810682222b0bd740746936525692d3909a54873b0168bda38e0cad4ec54dcac3fe77d20f418d3ec3f144868e40006fe142e43d6cec74863f159a13ccc7dcc8bf36c363c03b691ba47469c98aa5239d45fe6b1b379922a11c2191cb15907f8335bb69ebfa2af836efa418d4c5fb988e41c9c6cd2da2ec74f06e1d9ead45ad9a774a12c086be8e05b790c338'

# Get the absolute path of the text file
absolute_path = os.path.abspath(file_path)

# Get the directory path and file name separately
directory_path = os.path.dirname(absolute_path)
file_name = os.path.basename(absolute_path)

# Print the directory path and file name
print("Directory Path:", directory_path)
print("File Name:", file_name)


# In[ ]:


https://storage.googleapis.com/kagglesdsdata/datasets/9590/13660/fruit_data_with_colors.txt?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230709%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230709T051245Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=50c212e09a0ef321bdfaee371a91fe63387c4b165944c09445ec83f102be2124ffd5a44c961699078b48e8a1ca6d798e48fb75573c2ee73a703c21acdfb3ce9256b9441d2c07129e911f7bf14ebde0c296d7bf22dff67ef7b8b39826ee7f985822903bf0258eec655225c00814a9ffcef1654d20d34558d51ed60190dc93bdd7c3d78bed042677afe40db486298c447abf0210b60bf4b233ca35fec61e5a895449168216fac343d50c231daf0e5102115d763384d7057d7ba17a179140f4b043211064d56d60aa32c8b9e386f50b567d7d43677cf71d0e87c87ebea839cc67d7090604542442f627e714c2b5dc7407c0f608eed5b951b82a0364d32be97a1e4e


# In[10]:





# In[24]:


import pandas as pd
dataset_path=r"C:\Users\DELL\Downloads\fruit_data_with_colors.txt"
df=pd.read_csv(dataset_path,delimiter='\t')


# In[25]:


df


# In[26]:


df.size


# In[27]:


df.shape


# In[28]:


df.info()


# In[29]:


df[25:45]


# In[30]:


df.iloc[25:45,0:2]


# In[31]:


df.fruit_name.nunique()


# In[32]:


df.fruit_name.unique()


# In[33]:


df.groupby('fruit_name').size()


# In[ ]:




