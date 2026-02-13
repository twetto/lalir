use crate::matrix::{Matrix, Vector};
use std::fmt;

/// Error type for decomposition failures
#[derive(Debug)]
pub enum LinalgError {
    Singular,
    NotPositiveDefinite,
    RankDeficient,
}

impl fmt::Display for LinalgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinalgError::Singular => write!(f, "Matrix is singular"),
            LinalgError::NotPositiveDefinite => write!(f, "Matrix is not positive definite"),
            LinalgError::RankDeficient => write!(f, "Matrix is rank-deficient"),
        }
    }
}

/// Result of LU decomposition with partial pivoting: PA = LU
/// L (unit lower triangular) and U (upper triangular) are packed
/// into a single matrix, with L below the diagonal and U on/above.
pub struct LU<const N: usize> {
    /// Packed LU factors
    pub lu: Matrix<N, N>,
    /// Pivot indices
    pub piv: [usize; N],
}

impl<const N: usize> LU<N> {
    /// Solve Ax = b using the precomputed LU factorization.
    pub fn solve(&self, b: &Vector<N>) -> Vector<N> {
        // Apply permutation: y = Pb
        let mut y = Vector::<N>::zeros();
        for i in 0..N {
            y[i] = b[self.piv[i]];
        }

        // Forward substitution: solve Ly = Pb
        // L has implicit 1s on diagonal
        for i in 1..N {
            for j in 0..i {
                y[i] -= self.lu[(i, j)] * y[j];
            }
        }

        // Backward substitution: solve Ux = y
        let mut x = Vector::<N>::zeros();
        for i in (0..N).rev() {
            let mut sum = y[i];
            for j in (i + 1)..N {
                sum -= self.lu[(i, j)] * x[j];
            }
            x[i] = sum / self.lu[(i, i)];
        }

        x
    }
}

/// LU decomposition with partial pivoting.
/// Returns the packed LU matrix and pivot array, or an error if singular.
pub fn lu_decompose<const N: usize>(a: &Matrix<N, N>) -> Result<LU<N>, LinalgError> {
    let mut lu = *a;
    let mut piv = [0usize; N];
    for i in 0..N {
        piv[i] = i;
    }

    for k in 0..N {
        // Find pivot row
        let mut max_val = lu[(k, k)].abs();
        let mut max_row = k;
        for i in (k + 1)..N {
            let val = lu[(i, k)].abs();
            if val > max_val {
                max_val = val;
                max_row = i;
            }
        }

        if max_val < 1e-12 {
            return Err(LinalgError::Singular);
        }

        // Swap rows
        if max_row != k {
            for j in 0..N {
                let tmp = lu[(k, j)];
                lu[(k, j)] = lu[(max_row, j)];
                lu[(max_row, j)] = tmp;
            }
            piv.swap(k, max_row);
        }

        // Eliminate below pivot
        for i in (k + 1)..N {
            let f = lu[(i, k)] / lu[(k, k)];
            lu[(i, k)] = f; // Store multiplier
            for j in (k + 1)..N {
                lu[(i, j)] -= f * lu[(k, j)];
            }
        }
    }

    Ok(LU { lu, piv })
}
