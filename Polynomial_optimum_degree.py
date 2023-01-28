import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model 
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
df=pd.read_csv("C:/Users/Mahdieh/Desktop/Machine Learning/linear/FuelConsumption.csv")
x=df[["ENGINESIZE"]]
y=df[["CO2EMISSIONS"]]
mask=np.random.rand(len(df))<.8
train=df[mask]
test=df[~mask]
x_train=train[["ENGINESIZE"]]
y_train=train["CO2EMISSIONS"]
x_test=test[["ENGINESIZE"]]
y_test=test["CO2EMISSIONS"]
z=0
l=[]
for i in range (1,30):
    poly=PolynomialFeatures(degree=i)
    x_train_poly=poly.fit_transform(x_train)
    #print(x_train_poly)
    r=linear_model.LinearRegression()
    r.fit(x_train_poly,y_train)
    #print(r.coef_)
    #print(r.intercept_)
    xx=np.arange(1,10,.1)
    w=xx**i*r.coef_[i]+r.intercept_
    z=w+z
    x_test_poly=poly.fit_transform(x_test)
    y_test_predict=r.predict(x_test_poly)
    q=r2_score(y_test,y_test_predict)
    l.append(q)
print((l.index(max(l))+1))
#print(l)
t=np.arange(1,30,1)
plt.scatter(t,l)
plt.plot(t,l)
plt.xlabel("degree")
plt.ylabel("r2")
plt.legend()
plt.show()

