import numpy as np
import math as mt
import matplotlib.pyplot as plt  # Importa matplotlib

# Tu objeto original
objeto = np.array([
    [1, 1],
    [1, 3],
    [3, 3],
    [3,1]
])

def graficar_transformaciones(puntos_originales, puntos_transformacion,titulo):
    """
    Grafica dos conjuntos de puntos conectándolos para formar la figura completa.
    Se cierra el polígono añadiendo el primer punto al final.
    """
    plt.figure(figsize=(6, 6))
    
    # Cerramos el polígono para los puntos originales y transformados
    puntos_orig_cerrados = np.vstack([puntos_originales, puntos_originales[0]])
    puntos_transf_cerrados = np.vstack([puntos_transformacion, puntos_transformacion[0]])
    
    # Graficamos ambos conjuntos de puntos
    plt.plot(puntos_orig_cerrados[:, 0], puntos_orig_cerrados[:, 1], 'blue', linestyle='--', label='Original')
    plt.plot(puntos_transf_cerrados[:, 0], puntos_transf_cerrados[:, 1], 'red', label=titulo)
    
    plt.legend()
    plt.grid(True)
    plt.title('Transformaciones')
    plt.show()

def trasnlacion(objeto, dx, dy):
    desplazamiento = np.array([dx, dy])
    return objeto + desplazamiento

def escala(objeto, tx, ty):
    matriz_identidad = np.array([
        [tx, 0],
        [0, ty]
    ])
    return objeto @ matriz_identidad.T


def rotacion(objeto, grados):
    angulo = np.deg2rad(grados)
    matriz = np.array([
        [mt.cos(angulo), -mt.sin(angulo)],
        [mt.sin(angulo), mt.cos(angulo)]
    ])
    return objeto @ matriz.T



def reflejo(objeto,reflexion):
    
    match reflexion:
        
        case "x":
            matriz = np.array([
                [1,0],
                [0,-1]
            ])
            return objeto @ matriz.T    
        case "y":
            matriz = np.array([
                [-1,0],
                [0,1]
            ])
            return objeto @ matriz.T 


def skew(objeto, angulo):
    shear_factor = np.tan(np.deg2rad(angulo))  # Calcula el factor de skew
    matriz = np.array([
        [1, shear_factor],
        [shear_factor, 1]
    ])
    return objeto @ matriz.T

# Traslación
objeto_traslacion = trasnlacion(objeto, 4, 2)
graficar_transformaciones(objeto, objeto_traslacion,"Traslacion")

# Escala
objeto_escala = escala(objeto, 0.5, 0.5)
graficar_transformaciones(objeto, objeto_escala,"escala")

# Rotación
objeto_rotacion = rotacion(objeto, -15)
graficar_transformaciones(objeto, objeto_rotacion,"rotacion")

# Reflejo (por eje x, por ejemplo)
objeto_reflejo = reflejo(objeto, "x")
graficar_transformaciones(objeto, objeto_reflejo,"reflejo")

# Skew
objeto_skew = skew(objeto, 15)
graficar_transformaciones(objeto, objeto_skew,"skew")

# Combinacion de skew y de reflejo

objeto_skew = skew(objeto, 15)
objeto_skew_reflejo = reflejo(objeto_skew,"x")
graficar_transformaciones(objeto,objeto_skew_reflejo,"AGRUPACIONES")