#!/usr/bin/env python
# coding: utf-8

# # What is NumPy?
# 
# NumPy is a scientific computing package (library) for python programming language.
# 
# Numpy is a powerful Python programming language library to solve numerical problems.
# 
# 
# 
# 
# ##  What is the meaning of NumPy word?
# 
# Num stands for numerical and Py stands for Python programming language.
# 
# Python NumPy library is especially used for numeric and mathematical calculation like linear algebra, Fourier transform, and random number capabilities using Numpy array.
# 
# NumPy supports large data in the form of a multidimensional array (vector and matrix).
# 
# 
# 
# NumPy Python library is too simple to learn. The basic python programming and its other libraries like Pandas and Matplotlib knowledge will help to solve real-world problems.
# 
# 
# 
# 
# ## Why use NumPy for machine learning, Deep Learning, and Data Science?
# 
# To solve computer vision and MRI, etc. So for that machine learning model want to use images, but the ML model can’t read image directly. So need to convert image into numeric form and then fit into NumPy array. which is the best way to give data to the ML model.
# 
# 

# In[1]:


import numpy as np


# In[2]:


# 1d array
a=np.array([1,2,3,4])
print(a)
type(a)
print(a.ndim) # this a key  word to check  which dimension of array 1d 2d 3d


# In[3]:


# chek data type
a.dtype


# In[4]:


a=np.array([[1,2,3],[4,5,6],[1,2,3],[4,5,6]])
print(a)
a.ndim


# In[5]:


# 3dimesional array
a=np.array([[[1,2,3],[4,5,6]],[[1,6,8],[4,7,3]]])
print(a)


# In[6]:


# chek the size of array
a.size
# chek how many row and cloum
a.shape
# 4 row and 3 cloum


# In[8]:


# how to create a multiples row and coloums particualr ones  and zeros digit
# only ones and zeros perfrom not rest of numbers
a=np.ones(5) # 1 row and 5 cloumn
print(a)


# In[11]:


a=np.ones((5,3))
print(a)


# In[12]:


# convert in interger from 
a=np.ones((5,3),dtype=int)
print(a)


# In[13]:


# zeros
a=np.zeros((5,3),dtype=int)
print(a)


# In[14]:


# convert in boolen from 
a=np.ones ((5,3),dtype=bool)
print(a)


# In[15]:


# to create empty matrix 
# in empty matrix always genrated random numbers to change the row and column valus
a=np.empty((2,3))
print(a)


# In[16]:


# arange()
#a=np.arange(start,end,steps)
a=np.arange(1,13,2)
print(a) # odd number
b=np.arange(2,13,2)
print(b)# even numbers


# In[17]:


# linspace()
np.linspace(1,6,5) #  to find valves  between 1 to 6 


# In[18]:


## Reshape() 
ar_2d=a.reshape(2,3)
print(ar_2d)


# In[21]:


## revel this funcation can covert multidimnesional array in 1 d

a=np.array([[1,2,3],[4,5,6],[1,2,3],[4,5,6]])
print(a)
a.ravel()


# In[22]:


# tanspose() this funcation covert row to column
a.transpose()


# In[4]:


a1+a2 # addition


# # Mathamatical operation using in numpy

# In[3]:


# Mathamatical operation using in numpy
import numpy as np
a1=np.arange(1,10).reshape(3,3)
a2=np.arange(1,10).reshape(3,3)
print(a1)
print(a2)


# In[6]:


np.add(a1,a2)


# In[7]:


a1-a2 # (minus)


# In[8]:


a1/a2 # ( division)


# In[9]:


np.divide(a1,a2)


# In[10]:


a1*a2 # multiplications


# In[11]:


np.multiply(a1,a2)


# In[16]:


# to  perfrom matrix product multiplication like  1 st row ,1 column
a=a1@a2
print(a)


# In[20]:


#  maximum
a.max()
# index value of maximum number
a.argmax()


# In[22]:


# how to find maxium value in row  and column
# 0 coloum
#1 rows
a.max(axis=0)


# In[23]:


a.max(axis=1)


# In[24]:


a.min(axis=1)


# In[25]:


#  find sum of all value in matrix
np.sum(a)


# In[26]:


# find  a sum based on row and column
np.sum(a,axis=0) # row


# In[27]:


np.sum(a,axis=1)


# In[28]:


# mean 
np.mean(a)


# In[29]:


# squareroot
np.sqrt(a)


# In[30]:


# standared devision
np.std(a)


# In[33]:


# log
np.log(a)
np.log10(a)


# # Python Numpy array Silicing(:)
# 

# In[35]:


a=np.arange(1,101).reshape(10,10)
a


# In[36]:


# find 12  ex 1st row 1 column
a[1,1]


# In[37]:


# find 64 ex 6th row and 3rd column
a[6,3] 


# In[38]:


a[0]  # find single row


# In[41]:


a[2]


# In[39]:


# column find 
a[:,0]


# In[43]:


# change the column 2dimension row to column
a[:,0:1]


# In[46]:


a[:,1]


# In[48]:


a[:,1:2]


# In[49]:


a


# In[50]:


# to access particular column and row
a[0:4,0:4]


# In[53]:


a[:,1:3]


# In[54]:


a[:]


# In[55]:


#  find 2how many storange to requried  to store matrix
a.itemsize


# In[56]:


a.dtype


# In[57]:


32/8


# # Python Numpy array Conctination(join) and split

# In[62]:


a1=np.arange(1,17).reshape(4,4)
a1


# In[63]:


a2=np.arange(17,33).reshape(4,4)
a2


# In[70]:


# conctination array a1 and a2
a=np.concatenate((a1,a2))
a


# In[72]:


np.vstack((a1,a2))


# In[71]:


# conctination array a1 and a2 acc to row
np.concatenate((a1,a2),axis=1)


# In[73]:


np.hstack((a1,a2))


# In[74]:


a1


# In[79]:


# split a array
np.split(a1,2)


# In[84]:


k=np.split(a,2)
k


# In[85]:


type(k)


# In[91]:


d1=np.array([4,7,8,9])
np.split(d1,[1,3])


# # Find Triganometry sin() ,cos(),tan() using numpy funcation tignometry funcations

# In[92]:


import numpy as np
import matplotlib.pyplot as   plt


# In[98]:


np.sin(60)


# In[99]:


np.cos(180)


# In[100]:


np.tan(60)


# In[106]:


# find a sin on sin curve

x_sin=np.arange(0,3*np.pi,0.1)
print(x_sin)


# In[105]:


y_sin=np.sin(x_sin)
print(y_sin)


# In[107]:


plt.plot(x_sin,y_sin)
plt.show()
    


# # random sampling with numpy

# In[109]:


#how to genrate random number
import random


# In[114]:


np.random.random(1) # random value genrate a random number b/w  0to1


# In[121]:


np.random.random((3,3))


# In[161]:


# to genrate  int  random value  between 1 to 4
np.random.randint(1,4)


# In[183]:


np.random.randint(1,4,(4,4))


# In[192]:


np.random.randint(1,4,(3,4,4)) # 3d array create


# In[211]:


# with hlep of seed funcation we use same value
np.random.seed(5)
np.random.randint(1,4,(3,4,4))


# In[216]:


np.random.rand(3,3)


# In[217]:


# randn
np.random.randn(3,3) # give both postive and negative values 


# In[252]:


x=[1,55,6,8]
np.random.choice(x)


# # Numpy String Operations,comparison and information
# 

# In[258]:


# how to add 2 string
s1=" India Won Match"
s2=" opposite to pok"
np.char.add(s1,s2)


# In[259]:


np.char.upper(s2)


# In[255]:


# lower case 
np.char.lower(s1)


# In[267]:


np.char.center(s1,50) # to fit string into center


# In[269]:


np.char.center(s1,30)


# In[271]:


np.char.center(s1,30,fillchar="^")


# In[272]:


#split the string
np.char.split(s1)


# In[273]:


np.char.splitlines("hello\nindia")


# In[277]:


# replace 
s1= " jai"
s2=" harray"
np.char.replace(s1,"jai","ram")


# In[281]:


# equal
np.char.equal(s1,s2)


# In[280]:


# count
np.char.count(s2,"r")


# In[283]:


np.char.find(s2,"y") # 6 is index values

