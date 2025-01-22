import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo
dataframe = pd.read_csv("/Users/ferleon/Documents/GitHub/semestre_IV/machine_learning/tareas/tarea_1/df_hosp_apilado.csv", sep=',', header=0)

# Validar datos
print("Valores nulos por columna:")
print(dataframe.isnull().sum())

# Total por año
totales_por_year = dataframe.groupby('anio')['valor'].sum()
print("\nTotal de valores por año:")
print(totales_por_year)

# Filtrar indicador de interés
df_filtrado = dataframe[dataframe['indicador'] == 'Egresos Hospitalarios']

# Convertir 'fecha' a formato datetime y ordenar por fecha
df_filtrado.loc[:, 'fecha'] = pd.to_datetime(df_filtrado['fecha'])
df_filtrado = df_filtrado.sort_values(by='fecha')

# Graficar tendencia
plt.figure(figsize=(12, 6))
plt.plot(df_filtrado['fecha'], df_filtrado['valor'], marker='o')
plt.title('Tendencia de Egresos Hospitalarios', fontsize=14)
plt.xlabel('Fecha', fontsize=12)
plt.ylabel('Valor', fontsize=12)
plt.xticks(rotation=45)
plt.grid()
plt.show()