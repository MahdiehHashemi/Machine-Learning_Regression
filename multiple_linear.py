import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
df=pd.read_csv("C:/Users/Mahdieh/Desktop/Machine Learning/linear/FuelConsumption.csv")
#print(df.head(10))
#print(df["FUELCONSUMPTION_COMB"].head(10))
#print(df.describe())
cdf=df[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB","CO2EMISSIONS"]]
#cdf.hist()
#plt.show()
# plt.scatter(df["FUELCONSUMPTION_COMB"], df["CO2EMISSIONS"])
# plt.xlabel("FUELCONSUMPTION_COMB")
# plt.ylabel("CO2EMISSIONS")
# plt.show()
msk=np.random.rand(len(cdf))<0.8
train=cdf[msk]
#print(train.size)
test=cdf[~msk]
#print(test.size)
reg = linear_model.LinearRegression()
train_x=np.asanyarray(train[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB"]])
train_y=np.asanyarray(train[["CO2EMISSIONS"]])
reg.fit(train_x, train_y)
print("coefficient is: ", reg.coef_, " and intercept is: ", reg.intercept_)
y_=reg.predict(test[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB"]])
x=np.asanyarray(test[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB"]])
y=np.asanyarray(test[["CO2EMISSIONS"]])

print("residual sum of squares: ", np.mean(y-y_)**2)
print("R2-Score: ", r2_score(y, y_))
