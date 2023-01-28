#!/usr/bin/env python
# coding: utf-8

# In[106]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as pl


# In[107]:


df=pd.read_csv("C:/Users/Mahdieh/Desktop/Machine Learning/Excercise_house_price/house_price.csv")
df.head()


# In[90]:


df.describe()


# In[108]:


cdf=df[['Area','Room','Parking','Elevator','Address','Price','Warehouse','Price(USD)']]
cdf.head()


# In[109]:


df['Area'] = pd.to_numeric(df['Area'], errors='coerce')


# In[110]:


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
df.Parking= label.fit_transform(df.Parking)
df.head()


# In[111]:


cdf=df[['Area','Room','Parking','Elevator','Address','Price','Price(USD)']]
cdf.head(710)


# In[124]:


df=df.dropna()
df.head(709)


# In[113]:


plt.scatter(cdf['Area'],cdf['Price(USD)'],color='blue')
plt.xlabel('Area')
plt.ylabel('Price(USD)')
plt.show()


# In[114]:


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
df.Address = label.fit_transform(df.Address)
print(label.fit_transform(df.Address))


# In[115]:


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
df.Elevator = label.fit_transform(df.Elevator)


# In[116]:


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
df.Warehouse = label.fit_transform(df.Warehouse)


# In[117]:


cdf=df[['Area','Room','Parking','Elevator','Address','Price','Warehouse','Price(USD)']]
cdf.head(710)


# In[118]:


plt.scatter(cdf['Address'],cdf['Price(USD)'],color='blue')
plt.xlabel('Address')
plt.ylabel('Price(USD)')
plt.show()


# In[120]:


plt.scatter(cdf['Warehouse'],cdf['Price(USD)'],color='blue')
plt.show()


# In[121]:


plt.scatter(cdf['Address'],cdf['Price(USD)'],color='blue')
plt.show()


# In[266]:


msk=np.random.rand(len(df))<0.8
train=cdf[msk]
test=cdf[~msk]


# In[267]:


from sklearn import linear_model
regr=linear_model.LinearRegression()
x=np.asanyarray(train[['Area','Address','Warehouse']])
y=np.asanyarray(train['Price(USD)'])
regr.fit(x,y)
print('coefficients',regr.coef_)
print('intercept',regr.intercept_)


# In[268]:


y_hat=regr.predict(test[['Area','Address','Warehouse']])
x=np.asanyarray(test[['Area','Address','Warehouse']])
y=np.asanyarray(test[['Price(USD)']])
print("Rrsidual sum of squares: %.2f"
      % np.mean((y_hat-y)**2))
print('Variance score: %.2f' %regr.score(x,y))


# In[ ]:





# In[ ]:





# In[ ]:




