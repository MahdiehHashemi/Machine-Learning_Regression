import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import r2_score
df=pd.read_csv("C:/Users/Mahdieh/Desktop/Machine Learning/linear/FuelConsumption.csv")
cdf=df[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB","CO2EMISSIONS"]]
msk=np.random.rand(len(cdf))<0.8
train=cdf[msk]
test=cdf[~msk]
train_x=np.asanyarray(train[["ENGINESIZE"]])
train_y=np.asanyarray(train[["CO2EMISSIONS"]])
test_x=np.asanyarray(test[["ENGINESIZE"]])
test_y=np.asanyarray(test[["CO2EMISSIONS"]])
poly=PolynomialFeatures(degree=2)
train_x_poly=poly.fit_transform(train_x)
print(train_x[:3])
print(train_x_poly[:3])
reg=linear_model.LinearRegression()
reg.fit(train_x_poly, train_y)
print("coefficient is: ", reg.coef_, " and intercept is: ", reg.intercept_)
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
xx=np.arange(0,10.0,0.1)
yy=reg.intercept_[0]+reg.coef_[0][1]*xx+reg.coef_[0][2]*np.power(xx,2)
plt.plot(xx,yy,"-r")
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()
test_x_poly=poly.fit_transform(test_x)
test_y_=reg.predict(test_x_poly)
print("residual sum of squares: ", np.mean(test_y-test_y_)**2)
print("R2-Score: ", r2_score(test_y, test_y_))