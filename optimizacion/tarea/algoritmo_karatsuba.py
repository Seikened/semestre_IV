"""
DISCLAIMER:
Este código implementa aproximaciones matemáticas para calcular logaritmos naturales (ln)
y logaritmos en base 10 (log10) utilizando métodos como la serie de Taylor. 

Fue tomado y adaptado de una fuente externa con fines prácticos para realizar pruebas y
utilizarlo como si fuera parte de una biblioteca estándar. La intención es estudiar el 
comportamiento matemático de estos algoritmos, sin depender de funciones ya disponibles 
en las librerías estándar de Python, como math.log o math.log10.

Limitaciones:
- Este código es una aproximación y puede ser menos eficiente y menos preciso que 
  las funciones optimizadas disponibles en bibliotecas estándar.
- El número de términos en la serie de Taylor afecta directamente la precisión del resultado.
  Valores pequeños para `terms` pueden producir errores significativos.
- No es adecuado para números extremadamente grandes o pequeños debido a problemas de convergencia.

Responsabilidad:
- Este código se proporciona "como está", sin garantías de precisión absoluta.
- No debe usarse en aplicaciones críticas que dependan de cálculos precisos de logaritmos.

Propósito:
- Este código es únicamente para fines educativos y prácticos, para comprender 
  la implementación matemática de logaritmos.
"""


# Aproximación de ln(x) usando la serie de Taylor
def natural_log(x, terms=100):
    """
    Calcula el logaritmo natural (ln) de un número x > 0 utilizando una
    aproximación basada en la serie de Taylor.
    
    Args:
        x (float): Número positivo cuyo logaritmo natural será calculado.
        terms (int): Número de términos de la serie de Taylor para mejorar
                     la precisión del cálculo.
                     
    Returns:
        float: Aproximación del logaritmo natural de x.
    
    Raises:
        ValueError: Si x es menor o igual a 0, ya que ln(x) no está definido.
    """
    if x <= 0:
        raise ValueError("El logaritmo natural no está definido para valores menores o iguales a 0.")
    
    # Transformamos x al rango (0, 2] para mejorar la convergencia de la serie.
    # Esto lo hacemos dividiendo x por 2 repetidamente y contando las divisiones.
    n = 0
    while x > 2:
        x /= 2  # Reducimos x dividiéndolo entre 2.
        n += 1  # Contamos cuántas divisiones hicimos.

    # Calculamos ln(1 + z) con z = x - 1, usando la serie de Taylor:
    # ln(1 + z) = z - z^2/2 + z^3/3 - z^4/4 + ...
    z = x - 1
    ln = 0
    for i in range(1, terms + 1):
        term = ((-1) ** (i + 1)) * (z ** i) / i  # Cada término de la serie.
        ln += term  # Sumamos el término a la aproximación.

    # Ajustamos el logaritmo por las divisiones realizadas.
    # Cada división por 2 agrega ln(2) al resultado.
    return ln + n * 0.6931471805599453  # ln(2) ≈ 0.6931471805599453

# Logaritmo en base 10
def log10_custom(x, terms=100):
    """
    Calcula el logaritmo en base 10 de un número x > 0 utilizando
    la relación log10(x) = ln(x) / ln(10).
    
    Args:
        x (float): Número positivo cuyo logaritmo base 10 será calculado.
        terms (int): Número de términos de la serie de Taylor para el cálculo de ln(x).
                     
    Returns:
        float: Aproximación del logaritmo base 10 de x.
    """
    LN_10 = 2.302585092994046  # Constante: ln(10).
    return natural_log(x, terms) / LN_10  # Relación log10(x) = ln(x) / ln(10).


# Fin del código  de la biblioteca estándar


# ===============================================================================================================================================================================================================

"""
A partir de este punto el código es de mi autoría
"""



# Algoritmo Karatsuba
import time
import math

# Algoritmo Karatsuba
def contador_digitos(user_num):
    """
    Método iterativo para contar dígitos
    """
    num = user_num
    
    if num == 0:
        return 1
    elif num < 0:
        num = -num
    
    contador = 0
    while num > 0:
        num //= 10
        contador += 1
    return contador


def contador_digitos_opti(user_num):
    """
    Versión eficiente que utiliza logaritmo base 10 (implementación personalizada)
    """
    num = user_num
    
    if num == 0:
        return 1
    elif num < 0:
        num = -num
    
    return int(log10_custom(num)) + 1


def contador_digitos_std(user_num):
    """
    Versión utilizando math.log10 para contar los dígitos
    """
    num = user_num
    
    if num == 0:
        return 1
    elif num < 0:
        num = -num
    
    return int(math.log10(num)) + 1



import os
os.system("clear")
print("Esto es una prueba")
print("-"*50)
# Prueba y medición de tiempo
num_1 = 12345678901234567890

# Método iterativo
start_iter = time.perf_counter()
result_iter = contador_digitos(num_1)
end_iter = time.perf_counter()

# Método con logaritmo personalizado
start_opti = time.perf_counter()
result_opti = contador_digitos_opti(num_1)
end_opti = time.perf_counter()

# Método con biblioteca estándar
start_std = time.perf_counter()
result_std = contador_digitos_std(num_1)
end_std = time.perf_counter()

# Mostrar resultados y tiempos
print("\033[1;34m" + "Resultados del conteo de dígitos" + "\033[0m")
print("-" * 50)

print("\033[1;32m" + f"Método iterativo para {num_1}: {result_iter} dígitos" + "\033[0m")
print(f"Tiempo del método iterativo: {(end_iter - start_iter) * 1_000_000:.2f} microsegundos")

print("\033[1;32m" + f"Método eficiente (personalizado) para {num_1}: {result_opti} dígitos" + "\033[0m")
print(f"Tiempo del método eficiente (personalizado): {(end_opti - start_opti) * 1_000_000:.2f} microsegundos")

print("\033[1;32m" + f"Método con biblioteca estándar para {num_1}: {result_std} dígitos" + "\033[0m")
print(f"Tiempo del método estándar: {(end_std - start_std) * 1_000_000:.2f} microsegundos")

# Determinar el método más rápido
tiempos = {
    "Método iterativo": (end_iter - start_iter),
    "Método eficiente (personalizado)": (end_opti - start_opti),
    "Método con biblioteca estándar": (end_std - start_std)
}

ganador = min(tiempos, key=tiempos.get)
print("\033[1;33m" + f"El método más rápido es: {ganador}" + "\033[0m")

print("-" * 50)
print("\033[1;34m" + "La prueba terminó" + "\033[0m")
_ = input("Presiona Enter para continuar con el método Karatsuba")
os.system("clear")




def separador(num_1,num_2):
    
    if num_1 > num_2:
        mayor = num_1
    else:
        mayor = num_2
    
    tam_mayor = contador_digitos(mayor)
    
    mitad = (tam_mayor // 2) + (tam_mayor % 2)
    
    return mayor,tam_mayor,mitad


def nomalizador(num_user_1, num_user_2):
    """
    Divide los números en partes altas y bajas (am + b) * (cm + d).
    """
    num_1, num_2 = abs(num_user_1), abs(num_user_2)

    # Caso base: si ambos números tienen un solo dígito
    if contador_digitos(num_1) == 1 and contador_digitos(num_2) == 1:
        return num_1, num_2, 0, 0

    # Determinar el tamaño del mayor número
    mayor,tam_mayor,mitad = separador(num_1, num_2)
    
    m = tam_mayor // 2
    divisor = 10**m
    
    a = num_1 // divisor
    b = num_1 % divisor
    c = num_2 // divisor
    d = num_2 % divisor

    return a, c, b, d

def karatsuba(num_1, num_2):
    """
    Algoritmo de Karatsuba para multiplicación de números grandes.
    """
    # Caso base: multiplicación directa para un solo dígito
    if contador_digitos(num_1) == 1 and contador_digitos(num_2) == 1:
        return num_1 * num_2

    # Dividir los números en partes altas y bajas
    a, c, b, d = nomalizador(num_1, num_2)

    # Recursividad para las multiplicaciones clave
    ac = karatsuba(a, c)
    bd = karatsuba(b, d)
    terminos_cruzados = karatsuba(a + b, c + d) - ac - bd

    # Determinar potencias de 10 para combinar resultados
    mitad = max(contador_digitos(num_1), contador_digitos(num_2)) // 2
    mitad_2m = 10 ** (2 * mitad)
    mitad_m = 10 ** mitad

    # Combinar resultados finales
    return (ac * mitad_2m) + (terminos_cruzados * mitad_m) + bd


num_1 = 123454356783451234
num_2 = 678911563566784321

# Prueba y medición de tiempo para el algoritmo de Karatsuba
start_karatsuba = time.perf_counter()
resultado_karatsuba = karatsuba(num_1, num_2)
end_karatsuba = time.perf_counter()

# Prueba y medición de tiempo para la multiplicación normal
start_normal = time.perf_counter()
resultado_normal = num_1 * num_2
end_normal = time.perf_counter()

# Mostrar resultados y tiempos
print("\033[1;34m" + "Resultados de la multiplicación" + "\033[0m")
print("-" * 50)

print("\033[1;32m" + f"Resultado Karatsuba: {resultado_karatsuba}" + "\033[0m")
print(f"Tiempo del algoritmo Karatsuba: {(end_karatsuba - start_karatsuba) * 1_000_000:.2f} microsegundos")

print("\033[1;32m" + f"Resultado normal: {resultado_normal}" + "\033[0m")
print(f"Tiempo de la multiplicación normal: {(end_normal - start_normal) * 1_000_000:.2f} microsegundos")

# Comparar resultados
if resultado_karatsuba == resultado_normal:
    print("\033[1;33m" + "Son iguales" + "\033[0m")
else:
    print("\033[1;31m" + "No es igual" + "\033[0m")