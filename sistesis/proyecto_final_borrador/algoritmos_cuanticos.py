# algoritmos_cuanticos.py

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from tabulate import tabulate        # pip install tabulate
import matplotlib.pyplot as plt
import numpy as np
import time, qiskit

print(f"Versión de Qiskit: {qiskit.__version__}\n")

def medir_tiempo(func):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter_ns()
        res = func(*args, **kwargs)
        t1 = time.perf_counter_ns()
        # tiempo en microsegundos
        tiempo_us = round((t1 - t0) / 1_000, 3)
        return res, tiempo_us
    return wrapper

@medir_tiempo
def grover(n, target, shots=1024):
    qc = QuantumCircuit(n, n)
    # Superposición
    qc.h(range(n)); qc.barrier()
    # Oráculo
    for i, b in enumerate(target):
        if b == '0': qc.x(i)
    qc.h(n-1); qc.mcx(list(range(n-1)), n-1); qc.h(n-1)
    for i, b in enumerate(target):
        if b == '0': qc.x(i)
    qc.barrier()
    # Difusión
    qc.h(range(n)); qc.x(range(n))
    qc.h(n-1); qc.mcx(list(range(n-1)), n-1); qc.h(n-1)
    qc.x(range(n)); qc.h(range(n))
    # Medición
    qc.measure(range(n), range(n))
    # Simula
    sim = AerSimulator()
    job = sim.run(transpile(qc, sim), shots=shots)
    counts = job.result().get_counts()
    found = max(counts, key=counts.get)
    return {'found': found}

@medir_tiempo
def classical_search(n, target, repeats=1000):
    # Genero el espacio de búsqueda una sola vez
    space = [format(i, f'0{n}b') for i in range(2**n)]
    # Repito la búsqueda para acumular tiempo medible
    for _ in range(repeats):
        for state in space:
            if state == target:
                break
    return {'found': target, 'repeats': repeats}

def main():
    # Parámetros
    n = 4
    shots = 1024
    repeats = 1000
    search_space = [format(i, f'0{n}b') for i in range(2**n)]

    # Ejecutar y guardar resultados
    results = []
    for tgt in search_space:
        (q_res, q_time) = grover(n, tgt, shots)
        (c_res, c_time) = classical_search(n, tgt, repeats)
        results.append({
            'target': tgt,
            'grover_found': q_res['found'],
            'grover_time_us': q_time,
            'classical_found': c_res['found'],
            'classical_time_us': c_time
        })

    # Mostrar tabla
    headers = ['Target', 'Grover Found', 'Q-Time (µs)', 'Classical Found', 'C-Time (µs)']
    key_map = {
        'Target': 'target',
        'Grover Found': 'grover_found',
        'Q-Time (µs)': 'grover_time_us',
        'Classical Found': 'classical_found',
        'C-Time (µs)': 'classical_time_us'
    }
    table = [[r[key_map[h]] for h in headers] for r in results]
    print(tabulate(table, headers=headers, tablefmt='fancy_grid'))

    # Gráfica de barras agrupadas
    x = np.arange(len(results))
    q_times = [r['grover_time_us'] for r in results]
    c_times = [r['classical_time_us'] for r in results]
    width = 0.4

    fig, ax1 = plt.subplots(figsize=(12,6))
    ax2 = ax1.twinx()  # segundo eje Y

    # Grover en ax1
    ax1.bar(x - width/2, q_times, width, color='tab:blue', label='Grover')
    ax1.set_ylabel('Grover (µs)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Clásica en ax2
    ax2.bar(x + width/2, c_times, width, color='tab:orange', label='Clásica')
    ax2.set_ylabel('Clásica (µs)', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    ax1.set_xticks(x)
    ax1.set_xticklabels([r['target'] for r in results], rotation=45)
    ax1.set_xlabel('Estado objetivo')
    plt.title(f'Comparativa Grover vs Clásica (n={n}, repeats={repeats})')
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
