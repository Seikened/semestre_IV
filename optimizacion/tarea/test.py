def generar_primos():
    num = 2
    while True:  # Bucle infinito
        if all(num % i != 0 for i in range(2, int(num**0.5) + 1)):
            yield num  # Genera el siguiente primo
        num += 1

primos = generar_primos()
for _ in range(10):  # Imprime los primeros 10 números primos
    print(next(primos))