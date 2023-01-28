import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
df=pd.read_csv("C:/Users/Mahdieh/Desktop/Machine Learning/linear/china_gdp.csv")
#print(df.describe())
x_data, y_data=(df["Year"], df["Value"])
# plt.plot(x_data,y_data,"ro")
# plt.xlabel("Year")
# plt.ylabel("GDP")
#khodeman tabe ra bayad taarif konim 
def sigmoid(x,beta1,beta2):
    y=1/(1+np.exp(-beta1*(x-beta2)))
    return(y)
#######Blind Test########
b1=0.2
b2=1990.0
y_test_pred=sigmoid(x_data,b1,b2)
plt.plot(x_data,y_test_pred*15000000000000)
plt.plot(x_data,y_data,"ro")
plt.show()   
########Normalization#######
xdata=x_data/max(x_data)
ydata=y_data/max(y_data)
#########curve_fit##########
popt,pcov=curve_fit(sigmoid,xdata,ydata)
print("beta1= ",popt[0], " beta2= ",popt[1])
print (pcov) #it is a 282 array for our case with beta1 and beta2, and the diagonal numbers are related to the error of the popt
#print(np.sqrt(np.diag(pcov)))
print("R2 value is: "+ str(r2_score(y_data.values/max(y_data), sigmoid(x_data/max(x_data), popt[0], popt[1]))))
#########testing curve######
plt.plot(xdata,sigmoid(xdata, popt[0], popt[1]))
plt.plot(xdata,ydata,"ro")
plt.show()
#######important predict##### khodam neveshtam vali kheili engar khub nashod!!!
year= int(input("desired year "))
new_sigmoid=max(y_data)/(1+np.exp(-popt[0]*(year/max(x_data)-popt[1])))
print(new_sigmoid/max(y_data))
    
