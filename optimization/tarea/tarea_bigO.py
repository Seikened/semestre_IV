import numpy as np
import math as mt
import time
import json

archivo_json = "resultados_bigO.json"

tiempos = {
    "segundo": 10**6,
    "minuto": 60 * 10**6,
    "hora": 60 * 60 * 10**6,
    "dia": 24 * 60 * 60 * 10**6,
    "mes": 30 * 24 * 60 * 60 * 10**6,
    "anio": 12 * 30 * 24 * 60 * 60 * 10**6,
    "siglo": 100 * 12 * 30 * 24 * 60 * 60 * 10**6
}

try:
    with open(archivo_json, "r") as f:
        resultados_totales = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    resultados_totales = {}

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
    return factor, 0 

def n_ln(factor):
    t_i = time.perf_counter()
    resultado = factor * np.log(factor)
    t_f = time.perf_counter()
    return resultado, (t_f - t_i)

def n_2(factor):
    t_i = time.perf_counter()
    resultado = factor ** 2 
    t_f = time.perf_counter()
    return resultado, (t_f - t_i)

def n_3(factor):
    t_i = time.perf_counter()
    resultado = factor ** 3 
    t_f = time.perf_counter()
    return resultado, (t_f - t_i)

def dos_n(factor):
    t_i = time.perf_counter()
    resultado = 2 ** factor 
    t_f = time.perf_counter()
    return resultado, (t_f - t_i)

def factorial(factor):
    t_i = time.perf_counter()
    resultado = mt.factorial(factor) 
    t_f = time.perf_counter()
    return resultado, (t_f - t_i)

def prints(tiempos, name_prueba, prueba):
    if name_prueba not in resultados_totales:
        resultados_totales[name_prueba] = {}

    for label, tiempo in tiempos.items():
        valor, tiempo_ejecucion = prueba(tiempo)
        resultados_totales[name_prueba][label] = {
            "Valor": valor,
            "Tiempo de ejecución": f"{tiempo_ejecucion:.8f} segundos"
        }
    
    with open(archivo_json, "w") as f:
        json.dump(resultados_totales, f, indent=4, ensure_ascii=False)

    for label, datos in resultados_totales[name_prueba].items():
        print(f"{name_prueba}({label}): {datos['Valor']} | Tiempo: {datos['Tiempo de ejecución']}")

prints(tiempos, 'ln', ln)
prints(tiempos, 'raiz', raiz)
prints(tiempos, 'n', n)
prints(tiempos, 'n_ln', n_ln)
prints(tiempos, 'n_2', n_2)
prints(tiempos, 'n_3', n_3)
prints(tiempos, 'dos_n', dos_n)
prints(tiempos, 'factorial', factorial)

print("\n📜 Resultados guardados progresivamente en:", archivo_json)
