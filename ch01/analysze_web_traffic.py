# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 16:10:07 2017

@author: hungtran
"""

import scipy as sp
data = sp.genfromtxt("web_traffic.tsv",delimiter="\t")
x = data[:,0]
y = data[:,1]

x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.title("Web traffic over last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)], ['week %i' %w for w in range(10)])
plt.autoscale(tight=True)


week = 2
inflection = int(week*7*24) #calculate the inflection point in hours
'''
Separate data in to two sets
'''

x_1 = x[:inflection]
y_1 = y[:inflection]
x_2 = x[inflection:]
y_2 = y[inflection:]

def errors(f,x,y):
    return sp.sum((f(x)-y)**2)
#polyfit is used to find optimized model with a desired order of the polynomial (this case, 1)
#fp1 is parameters of the model
#residuals is error of the model
fp1, residuals, rank, sv, rcond = sp.polyfit(x_2,y_2,2,full=True)
print("Model parameters: %s" %fp1)
print residuals
#use poly1d to create model function from model parameters
f1 = sp.poly1d(fp1)
print(errors(f1,x_2,y_2))
#now, we use f1() to plot our trained model
fx = sp.linspace(x_2[0],x_2[-1],len(x_2)) #generate  x-values for plotting
plt.plot(fx,f1(fx),linewidth=2,color='blue')
plt.legend(['d=%i' %f1.order], loc='upper_left')

f_1 = sp.poly1d(sp.polyfit(x_1,y_1,1))
f_2 = sp.poly1d(sp.polyfit(x_2,y_2,1))
error1 = errors(f_1,x_1,y_1)
error2 = errors(f_2,x_2,y_2)
print ('Error inflection = %f' %(error1 + error2))

#now we plot 2 parts of model
fx1 = sp.linspace(0,x_1[-1],len(x_1))
plt.plot(fx1,f_1(fx1),'-.',linewidth=2,color='red')
plt.legend('Until week = %f' %week)
fx2 = sp.linspace(x_2[0],x_2[-1],len(x_2))
plt.plot(fx2,f_2(fx2),'--',linewidth=2,color='purple')
plt.legend('After weeks = %f' %week)
plt.grid()
plt.show()

week = 2
inflection = int(week*7*24)
x_3 = x[inflection:]
y_3 = y[inflection:]
fbt2 = sp.poly1d(sp.polyfit(x_3,y_3,2))
print (fbt2)
print (fbt2-100000)
from scipy.optimize import fsolve
reached_max = fsolve(fbt2-100000,800)/(7*24)
print("100,000 hits/hour expected at week %f" %reached_max[0])
