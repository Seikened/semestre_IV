def karatsuba_correcto(x, y):
    """
    Algoritmo de Karatsuba para multiplicación de números grandes.
    """
    # Caso base: multiplicación directa para un solo dígito
    if x < 10 or y < 10:
        return x * y

    # Determinar el tamaño de los números
    n = max(contador_digitos(x), contador_digitos(y))
    m = n // 2  # Mitad del tamaño

    # Dividir los números en partes altas y bajas
    high_x, low_x = divmod(x, 10 ** m)
    high_y, low_y = divmod(y, 10 ** m)

    # Recursividad para las multiplicaciones clave
    z0 = karatsuba_correcto(low_x, low_y)              # Parte baja
    z1 = karatsuba_correcto((low_x + high_x), (low_y + high_y))  # Parte cruzada
    z2 = karatsuba_correcto(high_x, high_y)            # Parte alta

    # Combinar los resultados finales
    return (z2 * 10 ** (2 * m)) + ((z1 - z2 - z0) * 10 ** m) + z0


def contador_digitos(num):
    """
    Contar el número de dígitos de un número.
    """
    if num == 0:
        return 1
    return len(str(abs(num)))


# Pruebas
num_1 = 12345
num_2 = 67891

resultado_karatsuba = karatsuba_correcto(num_1, num_2)
resultado_normal = num_1 * num_2

print(f"Resultado Karatsuba: {resultado_karatsuba}")
print(f"Resultado normal: {resultado_normal}")
