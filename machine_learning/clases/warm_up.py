import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


nombre_variables = ["longitud_sepalo", "ancho_sepalo", "longitud_petalo", "ancho_petalo", "clase"]

link = "https://raw.githubusercontent.com/ayrna/tutorial-scikit-learn-IMC/master/data/iris.csv"

iris = pd.read_csv(link, names=nombre_variables)

modIris = iris[['ancho_sepalo', 'longitud_sepalo']]

# Un valor de una fila y columna
print(iris.loc[7,'ancho_petalo'])

# Una columna
print(iris.loc[[1,10,20],['ancho_petalo', 'longitud_sepalo','clase']])

# Filas y columnas
print(iris.iloc[10:20,3:5])



iris_array = iris.values

print(iris_array[:,0])

print(iris_array[0:2,2:4])


print("PARTE 2")
colors = ['blue', 'red', 'green']
iris_target_names = np.unique(iris['clase'])


print(iris_target_names)



variable = 'longitud_petalo'

for nombre, color in zip(iris_target_names, colors):
    patrones = (iris['clase'] == nombre)
    plt.hist(iris.loc[patrones, variable], label=nombre, color=color)
    
plt.xlabel(variable)
plt.legend(loc='upper right')
plt.show()




variable_x = 'ancho_sepalo'
variable_y = 'longitud_sepalo'


for nombre,color in zip(iris_target_names,colors):
    patrones = (iris['clase'] == nombre)
    plt.scatter(iris.loc[patrones,variable_x],
                iris.loc[patrones,variable_y],
                label=nombre, c= color)

plt.xlabel(variable_x)
plt.ylabel(variable_y)
plt.legend(loc='upper left ')
plt.show()
