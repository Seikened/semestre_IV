import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


def msc(y,X,w):
    y_T = np.transpose(y)
    X_T = np.transpose(X)
    w_T = np.transpose(w)
    
    return (y_T*y) - (2*(y_T*X*w)) + (w_T*X_T*X*w)



X = np.array([1,2,3,4,5]).reshape(-1,1)
y = np.array([1.5, 1.7, 3.2, 3.8, 5.1])

model = LinearRegression()
model.fit(X,y)
y_pred = model.predict(X)
print(y_pred)


w_0 = model.intercept_
w = model.coef_

y_pred_ = w_0 + (X*w).reshape(1,-1)
print(y_pred_)





error_total = sum((y-y_pred)**2)/len(y)
print(error_total)

error_cuadratico = mean_squared_error(y, y_pred)
print(error_cuadratico)



grad = msc(y,X,w)
print(grad)