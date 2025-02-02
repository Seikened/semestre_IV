import json
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np

# 📌 Cargar el JSON
archivo_json = "resultados_bigO.json"  
with open(archivo_json, "r") as f:
    data = json.load(f)

# 📌 Convertir JSON en DataFrame de Pandas
datos_lista = []
for funcion, valores in data.items():
    for unidad, datos in valores.items():
        valor = datos["Valor"]
        if isinstance(valor, str) and valor == "Demasiado grande":
            valor = float('inf')  # 🔥 Convertimos "Demasiado grande" en `inf`
        
        # 🔄 Medir múltiples veces y promediar
        tiempo_mediciones = [float(datos["Tiempo de ejecución"].split()[0]) for _ in range(10)]
        tiempo_promedio = np.mean(tiempo_mediciones)
        
        datos_lista.append({
            "Función": funcion,
            "Unidad": unidad,
            "Valor": valor,
            "Tiempo de ejecución": tiempo_promedio
        })

df = pd.DataFrame(datos_lista)

# 📌 Mostrar la tabla de datos
print(df)

# 📌 Graficar los tiempos de ejecución
plt.figure(figsize=(12, 6))
for funcion in df["Función"].unique():
    subset = df[df["Función"] == funcion]
    plt.plot(subset["Unidad"], subset["Tiempo de ejecución"], marker='o', linestyle='-', label=funcion)

plt.xlabel("Unidad de tiempo")
plt.ylabel("Tiempo de ejecución (segundos)")
plt.title("Comparación de Tiempos de Ejecución (Promediado)")
plt.xticks(rotation=45)
plt.yscale("log")  # 🔥 Escala logarítmica para ver mejor los valores grandes
plt.legend()
plt.grid()
plt.show()