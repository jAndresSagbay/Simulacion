# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:57:10 2020

@author: usuario
"""
# Importar las librerias para el analasis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit
from numpy import poly1d,polyfit  
from sklearn.model_selection import train_test_split
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import math
import statistics
from scipy.integrate import odeint
from random import randrange # Obtener un numero randomico
import pygame
df=pd.read_csv('time-series-19-covid-combined_csv.csv')
df = df[df['Country/Region'].isin(['Mexico'])] #Filtro la Informacion solo para Ecuador
df = df.loc[:,['Date','Confirmed']] #Selecciono las columnas de analasis
# Expresar las fechas en numero de dias desde el 01 Enero
FMT = '%Y-%m-%d'
date = df['Date']
df['Date'] = date.map(lambda x : (datetime.strptime(x, FMT) - datetime.strptime("2020-01-01", FMT)).days)
df.plot(x ='Date', y='Confirmed')
x = list(df.iloc [:, 0]) # Fecha
y = list(df.iloc [:, 1]) # Numero de casos



#regresion lineal
lineal = linear_model.LinearRegression()
lineal.fit(np.array(x).reshape(-1,1), y)
plt.plot(x, y, 'x')
plt.plot(x, lineal.predict(np.array(x).reshape(-1, 1)))

j = 1
for i in range(39, 46):
    ##PREDICCION PARA 7 DIAS SIGUIENTES
    prediccion = lineal.predict([[i]])
    print(prediccion, 'Dia: ',j)
    j += 1
    


#regresion exponencial
def reg_exponencial(x, c0, c1, c2, c3):
    return c0 * np.exp(-c1 * x) + c2 + c3
g = [500, 0.3, 7, 0.3]
m = []
for i in range(len(x)):
    m.append(reg_exponencial(y[i], g[0], g[1], g[2], g[3]))
    
popt, pcov = curve_fit(reg_exponencial, x, m, g, maxfev=10000)
for i in range(len(x)):
    m[i] = reg_exponencial(x[i], popt[0], popt[1], popt[2], popt[3])
plt.plot(x, y)
plt.plot(x, list(reversed(m)))


#regresion polinomial
plt.plot(x, y, 'o')
p = poly1d(polyfit(x, y, deg=4))
Xtrain, Xtest, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(Xtrain, Xtest, y_train, y_test)
poli_reg = PolynomialFeatures(degree = 3)
X_train = poli_reg.fit_transform(np.array(Xtrain).reshape(-1,1))
X_test = poli_reg.fit_transform(np.array(Xtest).reshape(-1,1))
pr = linear_model.LinearRegression()
pr.fit(X_train, y_train)
Y_pred_pr = pr.predict(X_test)
#plt.plot(Xtest, y_test)
plt.plot(Xtest, Y_pred_pr, color='red')



#SIR
N = 250
#Cantidad de infectados iniciales
I0 = 1
#Cantidad de recuperados
R0 = 0
#Poblacion restante suceptibles
S0 = N -I0 - R0
print('antes')
def loss(point, data, S0, I0, R0):
    beta, gamma = point
    def SIR(t, y):
        S = y[0]
        I = y[1]
        R = y[2]
        return [-beta*S*I/N, beta*S*I/N-gamma*I, gamma*I]
        
    solve = solve_ivp(SIR, [0, len(data)], [S0,I0,R0], t_eval = np.arange(0, len(data), 1))
    r = math.sqrt(statistics.mean((solve.y[1] - data)**2))
    return r

opt = minimize(loss, [0.001, 0.001], args=(y, S0, I0, R0), method='L-BFGS-B', bounds=[(0.00000001, 0.4), (0.00000001, 0.4)])
beta, gamma = opt.x

t = np.linspace(2, 204)


def derivada(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N 
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


# Vector de condiciones iniciales
y0 = S0, I0, R0
# Integre las ecuaciones SIR en la cuadrícula de tiempo, t. A traves de la funcion odeint()
ret = odeint(derivada, y0, t, args=(N, beta, gamma))
S, I, R = ret.T # Obtenicion de resultados

#  Trace los datos en tres curvas separadas para S (t), I (t) y R (t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111,  axisbelow=True)
ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Sustible de infeccion')
ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infectados')
ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recuperados')
ax.set_xlabel('Tiempo en dias')
ax.set_ylabel('Numero de Personas')
ax.set_ylim(0,N*1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()
RE = ((beta/gamma)*N)
print('RE:', RE)

