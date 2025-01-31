use pyo3::prelude::*;
use num_bigint::BigInt;
use num_traits::{Zero, One};
use std::time::Instant;

/// Suma dos números grandes
#[pyfunction]
fn sum_bigint(a: &str, b: &str) -> PyResult<(String, u128)> {
    let a_bigint = a.parse::<BigInt>().map_err(|_| pyo3::exceptions::PyValueError::new_err("Número inválido"))?;
    let b_bigint = b.parse::<BigInt>().map_err(|_| pyo3::exceptions::PyValueError::new_err("Número inválido"))?;
    
    let start = Instant::now();
    let result = a_bigint + b_bigint;
    let elapsed = start.elapsed().as_nanos();
    
    Ok((result.to_string(), elapsed))
}

/// Multiplicación de números grandes
#[pyfunction]
fn multiply_bigint(a: &str, b: &str) -> PyResult<(String, u128)> {
    let a_bigint = a.parse::<BigInt>().map_err(|_| pyo3::exceptions::PyValueError::new_err("Número inválido"))?;
    let b_bigint = b.parse::<BigInt>().map_err(|_| pyo3::exceptions::PyValueError::new_err("Número inválido"))?;

    let start = Instant::now();
    let result = a_bigint * b_bigint;
    let elapsed = start.elapsed().as_nanos();
    
    Ok((result.to_string(), elapsed))
}

/// Exponenciación modular (a^b mod m)
#[pyfunction]
fn mod_exp(a: &str, b: &str, m: &str) -> PyResult<(String, u128)> {
    let a_bigint = a.parse::<BigInt>().map_err(|_| pyo3::exceptions::PyValueError::new_err("Número inválido"))?;
    let b_bigint = b.parse::<BigInt>().map_err(|_| pyo3::exceptions::PyValueError::new_err("Número inválido"))?;
    let m_bigint = m.parse::<BigInt>().map_err(|_| pyo3::exceptions::PyValueError::new_err("Número inválido"))?;

    let start = Instant::now();
    let result = a_bigint.modpow(&b_bigint, &m_bigint);
    let elapsed = start.elapsed().as_nanos();

    Ok((result.to_string(), elapsed))
}

/// Factorización de un número (buscamos el menor divisor primo)
#[pyfunction]
fn factorize(n: &str) -> PyResult<(String, u128)> {
    let n_bigint = n.parse::<BigInt>().map_err(|_| pyo3::exceptions::PyValueError::new_err("Número inválido"))?;

    let start = Instant::now();
    let mut factor = BigInt::from(2);
    while &factor * &factor <= n_bigint {
        if &n_bigint % &factor == BigInt::zero() {
            let elapsed = start.elapsed().as_nanos();
            return Ok((factor.to_string(), elapsed));
        }
        factor += BigInt::one();
    }
    let elapsed = start.elapsed().as_nanos();
    Ok((n_bigint.to_string(), elapsed))  // Si es primo, devolvemos el mismo número
}

#[pymodule]
fn rusty(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_bigint, py)?)?;
    m.add_function(wrap_pyfunction!(multiply_bigint, py)?)?;
    m.add_function(wrap_pyfunction!(mod_exp, py)?)?;
    m.add_function(wrap_pyfunction!(factorize, py)?)?;
    Ok(())
}
