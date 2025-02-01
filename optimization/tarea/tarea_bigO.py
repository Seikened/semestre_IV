import numpy as np
import math as mt
import time
import json

# Archivo donde se guardarán los resultados
archivo_json = "resultados_bigO.json"

# Tiempos
tiempos = {
    "segundo": 10**6,
    "minuto": 60 * 10**6,
    "hora": 60 * 60 * 10**6,
    "dia": 24 * 60 * 60 * 10**6,
    "mes": 30 * 24 * 60 * 60 * 10**6,
    "anio": 12 * 30 * 24 * 60 * 60 * 10**6,
    "siglo": 100 * 12 * 30 * 24 * 60 * 60 * 10**6
}

# Intentamos cargar resultados previos si existen
try:
    with open(archivo_json, "r") as f:
        resultados_totales = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    resultados_totales = {}

# Funciones de complejidad
def ln(factor):
    t_i = time.perf_counter()
    resultado = np.log(factor)
    t_f = time.perf_counter()
    return resultado, (t_f - t_i)

def raiz(factor):
    t_i = time.perf_counter()
    resultado = np.sqrt(factor)
    t_f = time.perf_counter()
    return resultado, (t_f - t_i)

def n(factor):
    return factor, 0  # No tiene procesamiento real

def n_ln(factor):
    t_i = time.perf_counter()
    resultado = factor * np.log(factor)
    t_f = time.perf_counter()
    return resultado, (t_f - t_i)

def n_2(factor):
    return factor ** 2

def n_3(factor):
    return factor ** 3 

def dos_n(factor):
    return 2 ** factor 

def factorial(factor):
    return mt.factorial(factor) 

# Función genérica para pruebas
def prints(tiempos, name_prueba, prueba):
    if name_prueba not in resultados_totales:
        resultados_totales[name_prueba] = {}

    for label, tiempo in tiempos.items():
        valor, tiempo_ejecucion = prueba(tiempo)
        resultados_totales[name_prueba][label] = {
            "Valor": valor,
            "Tiempo de ejecución": f"{tiempo_ejecucion:.8f} segundos"
        }
    
    # Guardar en JSON después de cada función
    with open(archivo_json, "w") as f:
        json.dump(resultados_totales, f, indent=4, ensure_ascii=False)

    # Imprimir resultados en consola
    for label, datos in resultados_totales[name_prueba].items():
        print(f"{name_prueba}({label}): {datos['Valor']} | Tiempo: {datos['Tiempo de ejecución']}")

# Ejecutar pruebas
prints(tiempos, 'ln', ln)
prints(tiempos, 'raiz', raiz)
prints(tiempos, 'n', n)
prints(tiempos, 'n_ln', n_ln)
prints(tiempos, 'n_2', n_2)
prints(tiempos, 'n_3', n_3)
prints(tiempos, 'dos_n', dos_n)
prints(tiempos, 'factorial', factorial)

print("\n📜 Resultados guardados progresivamente en:", archivo_json)