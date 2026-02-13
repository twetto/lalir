use crate::matrix::{Matrix, Vector};
use crate::lu::LinalgError;

// ============================================================
// QR result types
// ============================================================

/// QR decomposition result: A = Q * R
/// Q is MxM orthogonal, R is MxN upper triangular.
pub struct QR<const M: usize, const N: usize> {
    pub q: Matrix<M, M>,
    pub r: Matrix<M, N>,
}

/// Compact Householder QR: stores reflectors + betas for efficient solves
/// without forming Q explicitly.
pub struct HouseholderQR<const M: usize, const N: usize> {
    /// Packed matrix: R in upper triangle, reflector tails below diagonal
    pub packed: Matrix<M, N>,
    /// Householder betas
    pub beta: [f64; N],
}

/// QR with column pivoting: AP = QR
pub struct PivotedQR<const M: usize, const N: usize> {
    pub packed: Matrix<M, N>,
    pub beta: [f64; N],
    pub perm: [usize; N],
}

// ============================================================
// Gram-Schmidt QR
// ============================================================

/// QR decomposition via classical Gram-Schmidt.
/// Returns Q (MxM) and R (MxN), or error if rank-deficient.
/// Only works for M >= N (tall or square).
pub fn qr_gram_schmidt<const M: usize, const N: usize>(
    a: &Matrix<M, N>,
) -> Result<QR<M, N>, LinalgError> {
    let mut q_cols = *a; // will become Q's first N columns
    let mut r = Matrix::<M, N>::zeros(); // only top NxN is used

    for j in 0..N {
        // Compute norm of column j
        let mut norm_sq = 0.0;
        for i in 0..M {
            norm_sq += q_cols[(i, j)] * q_cols[(i, j)];
        }
        let rjj = norm_sq.sqrt();

        if rjj < 1e-12 {
            return Err(LinalgError::RankDeficient);
        }
        r[(j, j)] = rjj;

        // Normalize column j
        for i in 0..M {
            q_cols[(i, j)] /= rjj;
        }

        // Orthogonalize remaining columns against column j
        for k in (j + 1)..N {
            let mut dot = 0.0;
            for i in 0..M {
                dot += q_cols[(i, j)] * q_cols[(i, k)];
            }
            r[(j, k)] = dot;
            for i in 0..M {
                q_cols[(i, k)] -= dot * q_cols[(i, j)];
            }
        }
    }

    // Build full Q (MxM) by extending orthonormal basis
    let mut q = Matrix::<M, M>::zeros();
    // Copy computed columns
    for j in 0..N {
        for i in 0..M {
            q[(i, j)] = q_cols[(i, j)];
        }
    }
    // Extend with standard basis vectors, re-orthogonalize
    if M > N {
        extend_orthonormal_basis(&mut q, N);
    }

    Ok(QR { q, r })
}

/// Extend the first `rank` columns of Q to a full orthonormal basis
/// using modified Gram-Schmidt on candidate standard basis vectors.
fn extend_orthonormal_basis<const M: usize>(q: &mut Matrix<M, M>, rank: usize) {
    let mut col_count = rank;
    for candidate in 0..M {
        if col_count >= M {
            break;
        }
        // Start with e_candidate
        let mut v = [0.0f64; M];
        v[candidate] = 1.0;

        // Orthogonalize against existing columns
        for j in 0..col_count {
            let mut dot = 0.0;
            for i in 0..M {
                dot += q[(i, j)] * v[i];
            }
            for i in 0..M {
                v[i] -= dot * q[(i, j)];
            }
        }

        // Check if linearly independent
        let mut norm_sq = 0.0;
        for i in 0..M {
            norm_sq += v[i] * v[i];
        }
        let norm = norm_sq.sqrt();
        if norm < 1e-12 {
            continue; // Linearly dependent, skip
        }

        // Normalize and add as new column
        for i in 0..M {
            q[(i, col_count)] = v[i] / norm;
        }
        col_count += 1;
    }
}

// ============================================================
// Householder QR
// ============================================================

/// Generate a Householder vector v (with v[0] = 1) and beta such that
/// H = I - beta * v * v? zeros out elements below the first entry of x.
fn generate_householder(x: &[f64]) -> (Vec<f64>, f64) {
    let n = x.len();
    let mut v = x.to_vec();

    let norm_x = {
        let mut s = 0.0;
        for &xi in x.iter() {
            s += xi * xi;
        }
        s.sqrt()
    };

    if norm_x == 0.0 {
        return (vec![0.0; n], 0.0);
    }

    let alpha = if x[0] >= 0.0 { norm_x } else { -norm_x };
    v[0] += alpha;

    let scale = v[0];
    if scale == 0.0 {
        return (vec![0.0; n], 0.0);
    }

    for i in 1..n {
        v[i] /= scale;
    }
    v[0] = 1.0;

    // beta = 2 / (v^Tv)
    let mut vtv = 1.0; // v[0]^2 = 1
    for i in 1..n {
        vtv += v[i] * v[i];
    }

    (v, 2.0 / vtv)
}

/// QR decomposition via Householder reflections (compact storage).
/// Stores R in upper triangle and reflector tails below diagonal.
pub fn qr_householder<const M: usize, const N: usize>(
    a: &Matrix<M, N>,
) -> HouseholderQR<M, N> {
    let mut packed = *a;
    let mut beta = [0.0f64; N];

    for k in 0..N.min(M) {
        let len = M - k;

        // Extract column k from row k downward
        let mut x = vec![0.0; len];
        for i in 0..len {
            x[i] = packed[(k + i, k)];
        }

        // Generate reflector
        let (v, b) = generate_householder(&x);
        beta[k] = b;

        // Apply H to packed from (k,k) onward
        for j in k..N {
            let mut dot = 0.0;
            for i in 0..len {
                dot += v[i] * packed[(k + i, j)];
            }
            for i in 0..len {
                packed[(k + i, j)] -= b * v[i] * dot;
            }
        }

        // Store reflector tail below diagonal
        for i in 1..len {
            packed[(k + i, k)] = v[i];
        }
    }

    HouseholderQR { packed, beta }
}

impl<const M: usize, const N: usize> HouseholderQR<M, N> {
    /// Form the explicit Q matrix (MxM) from stored reflectors.
    pub fn q(&self) -> Matrix<M, M> {
        let mut q = Matrix::<M, M>::identity();

        for k in (0..N.min(M)).rev() {
            let len = M - k;
            let b = self.beta[k];

            // Reconstruct v
            let mut v = vec![0.0; len];
            v[0] = 1.0;
            for i in 1..len {
                v[i] = self.packed[(k + i, k)];
            }

            // Apply H = I - beta*vv^T to Q from the left
            for j in 0..M {
                let mut dot = 0.0;
                for i in 0..len {
                    dot += v[i] * q[(k + i, j)];
                }
                for i in 0..len {
                    q[(k + i, j)] -= b * v[i] * dot;
                }
            }
        }

        q
    }

    /// Extract R (upper triangular, MxN).
    pub fn r(&self) -> Matrix<M, N> {
        let mut r = Matrix::<M, N>::zeros();
        for i in 0..M {
            for j in i..N {
                r[(i, j)] = self.packed[(i, j)];
            }
        }
        r
    }

    /// Apply Q^T to a vector b without forming Q explicitly.
    pub fn qt_mul(&self, b: &Vector<M>) -> Vector<M> {
        let mut out = *b;

        for k in 0..N.min(M) {
            let len = M - k;
            let beta = self.beta[k];

            let mut v = vec![0.0; len];
            v[0] = 1.0;
            for i in 1..len {
                v[i] = self.packed[(k + i, k)];
            }

            let mut dot = 0.0;
            for i in 0..len {
                dot += v[i] * out[k + i];
            }
            for i in 0..len {
                out[k + i] -= beta * v[i] * dot;
            }
        }

        out
    }

    /// Solve Ax = b for square systems (NxN) using QR.
    pub fn solve(&self, b: &Vector<M>) -> Vector<N> {
        let qt_b = self.qt_mul(b);

        // Back substitution on the upper NxN part of R
        let mut x = Vector::<N>::zeros();
        for i in (0..N).rev() {
            let mut sum = qt_b[i];
            for j in (i + 1)..N {
                sum -= self.packed[(i, j)] * x[j];
            }
            x[i] = sum / self.packed[(i, i)];
        }

        x
    }

    /// Solve least-squares min ||Ax - b||_2 for overdetermined systems (M >= N).
    pub fn solve_least_squares(&self, b: &Vector<M>) -> Vector<N> {
        // Same as solve: Q^Tb then back-substitute on top N rows
        self.solve(b)
    }

    /// Check if the decomposed matrix is full rank.
    pub fn is_full_rank(&self, tol: f64) -> bool {
        for i in 0..N.min(M) {
            if self.packed[(i, i)].abs() < tol {
                return false;
            }
        }
        true
    }
}

// ============================================================
// Givens QR
// ============================================================

/// QR decomposition via Givens rotations.
/// Returns explicit Q and R.
pub fn qr_givens<const M: usize, const N: usize>(
    a: &Matrix<M, N>,
) -> QR<M, N> {
    let mut r = *a;
    let mut q = Matrix::<M, M>::identity();

    for j in 0..N {
        for i in (j + 1..M).rev() {
            let a_val = r[(i - 1, j)];
            let b_val = r[(i, j)];
            if b_val.abs() < 1e-12 {
                continue;
            }

            let h = a_val.hypot(b_val);
            let c = a_val / h;
            let s = -b_val / h;

            // Apply rotation to R (rows i-1, i)
            for col in 0..N {
                let r1 = r[(i - 1, col)];
                let r2 = r[(i, col)];
                r[(i - 1, col)] = c * r1 - s * r2;
                r[(i, col)] = s * r1 + c * r2;
            }

            // Apply rotation to Q columns (accumulate Q^T, then transpose)
            for col in 0..M {
                let q1 = q[(i - 1, col)];
                let q2 = q[(i, col)];
                q[(i - 1, col)] = c * q1 - s * q2;
                q[(i, col)] = s * q1 + c * q2;
            }
        }
    }

    // Q was accumulated as Q?, transpose it
    q = q.transpose();

    QR { q, r: Matrix::<M, N>::zeros().apply(|out| {
        for i in 0..M {
            for j in 0..N {
                out[(i, j)] = r[(i, j)];
            }
        }
    })}
}

// Helper to avoid returning r directly when types might differ
impl<const M: usize, const N: usize> Matrix<M, N> {
    /// Apply a closure for in-place initialization, then return self.
    fn apply(mut self, f: impl FnOnce(&mut Self)) -> Self {
        f(&mut self);
        self
    }
}

// ============================================================
// QR with Column Pivoting (QRCP)
// ============================================================

/// QR decomposition with column pivoting: A * P = Q * R
pub fn qr_col_pivoted<const M: usize, const N: usize>(
    a: &Matrix<M, N>,
) -> PivotedQR<M, N> {
    let mut packed = *a;
    let mut beta = [0.0f64; N];
    let mut perm = [0usize; N];
    for j in 0..N {
        perm[j] = j;
    }

    // Column norm cache
    let mut col_norms = [0.0f64; N];
    for j in 0..N {
        let mut sum = 0.0;
        for i in 0..M {
            sum += packed[(i, j)] * packed[(i, j)];
        }
        col_norms[j] = sum;
    }

    for k in 0..N.min(M) {
        // Find pivot column
        let mut pivot = k;
        let mut max_norm = col_norms[k];
        for j in (k + 1)..N {
            if col_norms[j] > max_norm {
                max_norm = col_norms[j];
                pivot = j;
            }
        }

        // Swap columns
        if pivot != k {
            for i in 0..M {
                let tmp = packed[(i, k)];
                packed[(i, k)] = packed[(i, pivot)];
                packed[(i, pivot)] = tmp;
            }
            perm.swap(k, pivot);
            col_norms.swap(k, pivot);
        }

        // Householder on column k
        let len = M - k;
        let mut x = vec![0.0; len];
        for i in 0..len {
            x[i] = packed[(k + i, k)];
        }

        let (v, b) = generate_householder(&x);
        beta[k] = b;

        for j in k..N {
            let mut dot = 0.0;
            for i in 0..len {
                dot += v[i] * packed[(k + i, j)];
            }
            for i in 0..len {
                packed[(k + i, j)] -= b * v[i] * dot;
            }
        }

        // Store reflector tail
        for i in 1..len {
            packed[(k + i, k)] = v[i];
        }

        // Update column norms
        for j in (k + 1)..N {
            let mut sum = 0.0;
            for i in (k + 1)..M {
                sum += packed[(i, j)] * packed[(i, j)];
            }
            col_norms[j] = sum;
        }
    }

    PivotedQR { packed, beta, perm }
}

impl<const M: usize, const N: usize> PivotedQR<M, N> {
    /// Solve least-squares min ||Ax - b||? with column pivoting.
    pub fn solve_least_squares(&self, b: &Vector<M>) -> Vector<N> {
        // Apply Q^T to b
        let mut qt_b = *b;
        for k in 0..N.min(M) {
            let len = M - k;
            let beta = self.beta[k];

            let mut v = vec![0.0; len];
            v[0] = 1.0;
            for i in 1..len {
                v[i] = self.packed[(k + i, k)];
            }

            let mut dot = 0.0;
            for i in 0..len {
                dot += v[i] * qt_b[k + i];
            }
            for i in 0..len {
                qt_b[k + i] -= beta * v[i] * dot;
            }
        }

        // Back substitution
        let mut x_perm = Vector::<N>::zeros();
        for i in (0..N).rev() {
            let mut sum = qt_b[i];
            for j in (i + 1)..N {
                sum -= self.packed[(i, j)] * x_perm[j];
            }
            x_perm[i] = sum / self.packed[(i, i)];
        }

        // Undo permutation: x[perm[i]] = x_perm[i]
        let mut x = Vector::<N>::zeros();
        for i in 0..N {
            x[self.perm[i]] = x_perm[i];
        }

        x
    }
}
