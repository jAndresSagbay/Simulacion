# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 10:52:21 2020

@author: usuario
"""

import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'

df = pd.read_csv(url).fillna(0)
df