import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
df=pd.read_csv("C:/Users/Mahdieh/Desktop/Machine Learning/linear/FuelConsumption.csv")
print(df["MODEL"].head())
print(df.describe())
#print(df.head(10))
#print(df["FUELCONSUMPTION_COMB"].head(10))
cdf=df[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB","CO2EMISSIONS"]]
print(cdf.head(9))
#cdf.hist()
#plt.show()
#plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color="blue")
#plt.xlabel("FUELCONSUMPTION_COMB")
#plt.ylabel("CO2EMISSIONS")
#plt.show()
#plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color="blue")
#plt.xlabel("ENGINESIZE")
#plt.ylabel("CO2EMISSIONS")
#plt.show()
print(len(cdf))
msk=np.random.rand(len(cdf))<0.8
train=cdf[msk]
#print(len(train))
#print(train.size)
test=cdf[~msk]
#print(test.size)
fig=plt.figure()
ax1=fig.add_subplot(111)
ax1.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color="blue")
ax1.scatter(test.ENGINESIZE, test.CO2EMISSIONS, color="red")
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()
############################
reg = linear_model.LinearRegression()
train_x=np.asanyarray(train[["ENGINESIZE"]])
train_y=np.asanyarray(train[["CO2EMISSIONS"]])
reg.fit(train_x, train_y)
print("coefficient is: ", reg.coef_, " and intercept is: ", reg.intercept_)
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color="blue")
plt.plot(train_x, reg.coef_[0][0]*train_x+reg.intercept_[0],color="red")
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()
test_x=np.asanyarray(test[["ENGINESIZE"]])
test_y=np.asanyarray(test[["CO2EMISSIONS"]])
print(reg.predict([[2.5]]))
test_y_=reg.predict(test_x)
print("Mean absoulte error: ", np.mean(np.absolute(test_y-test_y_)))
print("Residual sum of squares (MSE): ", np.mean((test_y_-test_y)**2))
print("R2-Score: ", r2_score(test_y, test_y_))
