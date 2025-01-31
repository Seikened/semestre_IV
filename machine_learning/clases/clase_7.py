import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
import pandas as pd



ruta = '/Users/ferleon/Documents/GitHub/semestre_IV/machine_learning/clases/USA_Housing.csv'

dataframe = pd.read_csv(ruta, header=0)

print(dataframe.shape)





X = np.array([1,2,3,4,5]).reshape(-1,1)
y = np.array([1.5, 1.7, 3.2, 3.8, 5.1])


model = SGDRegressor(alpha=0.0001, max_iter=1000, penalty=None)

model.fit(X,y)
y_pred = model.predict(X)

w_0 = model.intercept_
w = model.coef_
n_iter = model.n_iter