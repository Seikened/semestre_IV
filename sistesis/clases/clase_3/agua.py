import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ruta = '/Users/ferleon/Documents/GitHub/semestre_IV/sistesis/clases/clase_3/Disponibilidad_del_Agua_en_Le_n__Corregido_.csv'

dataset = pd.read_csv(ruta, sep=',', encoding='utf-8', header=0)

print(dataset.head())

# Cargar el dataset
df = pd.read_csv(ruta, sep=',', encoding='utf-8', header=0)

# Configuración de estilo
df.set_index("Año", inplace=True)
sns.set(style="whitegrid")

# Estadística descriptiva
print("\nEstadísticas descriptivas:")
print(df.describe())

# Gráfico 1: Evolución de la Producción de Agua
df["Producción (millones de m3)"].plot(figsize=(10,5), marker='o', linestyle='-', color='blue')
plt.title("Evolución de la Producción de Agua en León (1989 - 2022)")
plt.xlabel("Año")
plt.ylabel("Millones de m³")
plt.grid()
plt.show()

# Gráfico 2: Correlación entre Población y Producción de Agua
plt.figure(figsize=(8,6))
sns.scatterplot(x=df["Población (millones de habitantes)"], y=df["Producción (millones de m3)"], hue=df.index, palette="coolwarm")
plt.title("Relación entre Población y Producción de Agua")
plt.xlabel("Población (millones de habitantes)")
plt.ylabel("Producción de agua (millones de m³)")
plt.grid()
plt.show()

# Gráfico 3: Evolución de la Eficiencia Física
df["Eficiencia Física (%)"].plot(figsize=(10,5), marker='s', linestyle='-', color='green')
plt.title("Evolución de la Eficiencia Física del Sistema de Agua")
plt.xlabel("Año")
plt.ylabel("Eficiencia (%)")
plt.grid()
plt.show()

# Gráfico 4: Relación entre Dotación de Agua y Población
plt.figure(figsize=(8,6))
sns.scatterplot(x=df["Población (millones de habitantes)"], y=df["Dotación (millones de m3)"], hue=df.index, palette="viridis")
plt.title("Dotación de Agua vs Crecimiento Poblacional")
plt.xlabel("Población (millones de habitantes)")
plt.ylabel("Dotación de agua (millones de m³)")
plt.grid()
plt.show()

# Gráfico 5: Histogramas de las variables
for column in df.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column], kde=True, bins=10, color='skyblue')
    plt.title(f"Distribución de {column}")
    plt.xlabel(column)
    plt.ylabel("Frecuencia")
    plt.grid()
    plt.show()

# Gráfico 6: Boxplots para detectar valores atípicos
for column in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(y=df[column], color='orange')
    plt.title(f"Valores atípicos en {column}")
    plt.ylabel(column)
    plt.grid()
    plt.show()

# Análisis de correlación
correlacion = df.corr()
print("\nCorrelaciones entre variables:")
print(correlacion)