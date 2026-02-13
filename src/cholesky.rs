use crate::matrix::{Matrix, Vector};
use crate::lu::LinalgError;

/// Result of Cholesky decomposition: A = L * L^T
/// L is lower triangular.
pub struct Cholesky<const N: usize> {
    pub l: Matrix<N, N>,
}

impl<const N: usize> Cholesky<N> {
    /// Solve Ax = b via forward substitution (Ly = b) then back substitution (L^Tx = y).
    pub fn solve(&self, b: &Vector<N>) -> Vector<N> {
        // Forward substitution: Ly = b
        let mut y = Vector::<N>::zeros();
        for i in 0..N {
            let mut sum = b[i];
            for j in 0..i {
                sum -= self.l[(i, j)] * y[j];
            }
            y[i] = sum / self.l[(i, i)];
        }

        // Backward substitution: L^Tx = y
        let mut x = Vector::<N>::zeros();
        for i in (0..N).rev() {
            let mut sum = y[i];
            for j in (i + 1)..N {
                sum -= self.l[(j, i)] * x[j]; // L^T[i][j] = L[j][i]
            }
            x[i] = sum / self.l[(i, i)];
        }

        x
    }
}

/// Cholesky decomposition for symmetric positive definite matrices.
/// Returns L such that A = L * L?, or error if not SPD.
pub fn cholesky_decompose<const N: usize>(a: &Matrix<N, N>) -> Result<Cholesky<N>, LinalgError> {
    let mut l = *a;

    for i in 0..N {
        for j in 0..=i {
            let mut sum = l[(i, j)];
            for k in 0..j {
                sum -= l[(i, k)] * l[(j, k)];
            }

            if i == j {
                if sum <= 0.0 {
                    return Err(LinalgError::NotPositiveDefinite);
                }
                l[(i, j)] = sum.sqrt();
            } else {
                l[(i, j)] = sum / l[(j, j)];
            }
        }

        // Zero upper triangle
        for j in (i + 1)..N {
            l[(i, j)] = 0.0;
        }
    }

    Ok(Cholesky { l })
}

/// Result of LDL? decomposition: A = L * D * L^T
/// L is unit lower triangular, D is diagonal (stored as a vector).
pub struct LDLT<const N: usize> {
    pub l: Matrix<N, N>,
    pub d: [f64; N],
}

impl<const N: usize> LDLT<N> {
    /// Solve Ax = b via three steps:
    /// 1. Ly = b      (forward, L is unit lower triangular)
    /// 2. Dz = y      (diagonal)
    /// 3. L?x = z     (backward)
    pub fn solve(&self, b: &Vector<N>) -> Vector<N> {
        // Step 1: Ly = b (L has implicit 1s on diagonal)
        let mut y = Vector::<N>::zeros();
        for i in 0..N {
            let mut sum = b[i];
            for j in 0..i {
                sum -= self.l[(i, j)] * y[j];
            }
            y[i] = sum;
        }

        // Step 2: Dz = y
        let mut z = Vector::<N>::zeros();
        for i in 0..N {
            z[i] = y[i] / self.d[i];
        }

        // Step 3: L^Tx = z
        let mut x = Vector::<N>::zeros();
        for i in (0..N).rev() {
            let mut sum = z[i];
            for j in (i + 1)..N {
                sum -= self.l[(j, i)] * x[j]; // L^T[i][j] = L[j][i]
            }
            x[i] = sum;
        }

        x
    }
}

/// LDL? decomposition for symmetric matrices.
/// Returns L (unit lower triangular) and D (diagonal), or error if singular.
pub fn ldlt_decompose<const N: usize>(a: &Matrix<N, N>) -> Result<LDLT<N>, LinalgError> {
    let mut l = *a;
    let mut d = [0.0f64; N];

    for j in 0..N {
        // Compute D[j]
        let mut sum = l[(j, j)];
        for k in 0..j {
            sum -= l[(j, k)] * l[(j, k)] * d[k];
        }
        d[j] = sum;

        if d[j].abs() < 1e-12 {
            return Err(LinalgError::Singular);
        }

        // Compute L[i][j] for i > j
        for i in (j + 1)..N {
            let mut val = l[(i, j)];
            for k in 0..j {
                val -= l[(i, k)] * l[(j, k)] * d[k];
            }
            l[(i, j)] = val / d[j];
        }
    }

    // Set L to unit lower triangular (1s on diagonal, 0s above)
    for i in 0..N {
        l[(i, i)] = 1.0;
        for j in (i + 1)..N {
            l[(i, j)] = 0.0;
        }
    }

    Ok(LDLT { l, d })
}
