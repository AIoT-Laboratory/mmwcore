//! Small Hermitian matrices; diagonal loading scales with measured mean power.
use num_complex::Complex64;

pub(super) fn covariance(samples: &[Complex64], size: usize) -> Vec<Complex64> {
    let mut result = vec![Complex64::default(); size * size];
    for x in samples.chunks_exact(size) {
        for i in 0..size {
            for j in 0..=i {
                result[i * size + j] += x[i] * x[j].conj();
            }
        }
    }
    for i in 0..size {
        for j in 0..=i {
            result[i * size + j] /= (samples.len() / size) as f64;
            result[j * size + i] = result[i * size + j].conj();
        }
    }
    result
}

pub(super) fn multiply(matrix: &[Complex64], vector: &[Complex64]) -> Vec<Complex64> {
    matrix
        .chunks_exact(vector.len())
        .map(|row| row.iter().zip(vector).map(|(m, v)| m * v).sum())
        .collect()
}

pub(super) fn loaded_inverse(
    matrix: &[Complex64],
    size: usize,
    gamma: f64,
) -> Result<Option<Vec<Complex64>>, String> {
    let scale = (0..size).map(|i| matrix[i * size + i].re).sum::<f64>() / size as f64;
    if scale == 0. {
        return Ok(None);
    }
    let mut lower = vec![Complex64::default(); size * size];
    for i in 0..size {
        for j in 0..=i {
            let mut value = matrix[i * size + j] / scale;
            if i == j {
                value.re += gamma;
            }
            for k in 0..j {
                value -= lower[i * size + k] * lower[j * size + k].conj();
            }
            lower[i * size + j] = if i == j {
                if !value.re.is_finite() || value.re <= 0. {
                    return Err("Loaded Capon covariance is not positive definite".into());
                }
                Complex64::new(value.re.sqrt(), 0.)
            } else {
                value / lower[j * size + j].re
            };
        }
    }
    let mut inverse = vec![Complex64::default(); size * size];
    for column in 0..size {
        let mut y = vec![Complex64::default(); size];
        for i in 0..size {
            let mut value = Complex64::new(f64::from(i == column), 0.);
            for j in 0..i {
                value -= lower[i * size + j] * y[j];
            }
            y[i] = value / lower[i * size + i].re;
        }
        let mut x = vec![Complex64::default(); size];
        for i in (0..size).rev() {
            let mut value = y[i];
            for j in i + 1..size {
                value -= lower[j * size + i].conj() * x[j];
            }
            x[i] = value / lower[i * size + i].re;
            inverse[i * size + column] = x[i] / scale;
        }
    }
    Ok(Some(inverse))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complex_inverse_has_correct_conjugation_and_scale() {
        let a = vec![
            Complex64::new(4., 0.),
            Complex64::new(1., 2.),
            Complex64::new(1., -2.),
            Complex64::new(3., 0.),
        ];
        let inv = loaded_inverse(&a, 2, 0.03).unwrap().unwrap();
        let mut loaded = a;
        loaded[0].re += 0.105;
        loaded[3].re += 0.105;
        for column in 0..2 {
            let product = multiply(&loaded, &[inv[column], inv[2 + column]]);
            for (row, value) in product.iter().enumerate() {
                assert!((*value - Complex64::new(f64::from(row == column), 0.)).norm() < 1e-12);
            }
        }
        assert!(
            loaded_inverse(&[Complex64::default(); 4], 2, 0.03)
                .unwrap()
                .is_none()
        );
    }
}
