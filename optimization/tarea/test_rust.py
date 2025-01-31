import rusty
import time

def python_sum_bigint(a: str, b: str):
    # 1) Parseamos antes de cronometrar la operación
    a_int = int(a)
    b_int = int(b)
    # 2) Medimos solo la operación
    start = time.perf_counter()
    result = a_int + b_int
    end = time.perf_counter()
    op_time_ns = (end - start) * 1e9
    return result, op_time_ns

def python_multiply_bigint(a: str, b: str):
    a_int = int(a)
    b_int = int(b)
    start = time.perf_counter()
    result = a_int * b_int
    end = time.perf_counter()
    op_time_ns = (end - start) * 1e9
    return result, op_time_ns

def python_mod_exp(a: str, b: str, m: str):
    a_int = int(a)
    b_int = int(b)
    m_int = int(m)
    start = time.perf_counter()
    result = pow(a_int, b_int, m_int)
    end = time.perf_counter()
    op_time_ns = (end - start) * 1e9
    return result, op_time_ns

def python_factorize(n: str):
    n_int = int(n)
    start = time.perf_counter()
    # Buscamos menor divisor
    factor = None
    for i in range(2, int(n_int**0.5) + 1):
        if n_int % i == 0:
            factor = i
            break
    # Si no hay divisor, el número es primo
    result = factor if factor else n_int
    end = time.perf_counter()
    op_time_ns = (end - start) * 1e9
    return result, op_time_ns

def benchmark_rust_vs_python(operation_name, rust_func, py_func, *args):
    """Compara el tiempo de operación pura (sin parseo) en Rust vs Python."""
    print(f"\n🔹 **Probando {operation_name}**")

    # ——— RUST ———
    start_rust = time.perf_counter()
    rust_result, rust_time_op = rust_func(*args)  # (str, u128 ns)
    end_rust = time.perf_counter()
    rust_total_ns = (end_rust - start_rust) * 1e9  # tiempo total en la llamada Rust (incl. parseo en Rust)
    
    print(f"✅ Resultado en Rust: {rust_result[:12]}...")  
    print(f"⏱️ Tiempo en Rust (solo operación): {rust_time_op} ns | {rust_total_ns:.0f} ns (llamada completa)")

    # ——— PYTHON ———
    start_py = time.perf_counter()
    python_result, py_time_op = py_func(*args)
    end_py = time.perf_counter()
    py_total_ns = (end_py - start_py) * 1e9

    print(f"✅ Resultado en Python: {str(python_result)[:12]}...")
    print(f"⏱️ Tiempo en Python (solo operación): {py_time_op:.0f} ns | {py_total_ns:.0f} ns (llamada completa)")

    # ——— Comparación de operación pura ———
    if rust_time_op < py_time_op:
        print(f"🏆 **Rust fue {py_time_op / rust_time_op:.2f} veces más rápido que Python** (operación pura)")
    else:
        print(f"🐍 **Python fue {rust_time_op / py_time_op:.2f} veces más rápido que Rust** (operación pura)")

# --------------
# EJEMPLO DE USO
# --------------
if __name__ == "__main__":
    a, b = str(10**50), str(10**50)
    modulo = str(10**20)

    benchmark_rust_vs_python(
        "Suma de enteros grandes",
        rusty.sum_bigint,
        python_sum_bigint,
        a, b
    )

    benchmark_rust_vs_python(
        "Multiplicación de enteros grandes",
        rusty.multiply_bigint,
        python_multiply_bigint,
        a, b
    )

    benchmark_rust_vs_python(
        "Exponenciación modular",
        rusty.mod_exp,
        python_mod_exp,
        a, b, modulo
    )

    benchmark_rust_vs_python(
        "Factorización",
        rusty.factorize,
        python_factorize,
        a
    )