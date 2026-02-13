use crate::matrix::Matrix;
use crate::svd::SVD;

const MAX_ITER: usize = 100;
const EPS: f64 = 1e-12;

// ============================================================
// Helpers
// ============================================================

/// Givens rotation: returns (c, s) such that
///   [c  s] * [a] = [r]
///   [-s c]   [b]   [0]
fn givens(a: f64, b: f64) -> (f64, f64) {
    if b.abs() < 1e-15 {
        return (1.0, 0.0);
    }
    let r = a.hypot(b);
    (a / r, b / r)
}

/// Apply Givens rotation to two rows (r1, r2) of a matrix stored as 2D array.
/// new_r1 =  c * row_r1 + s * row_r2
/// new_r2 = -s * row_r1 + c * row_r2
fn rotate_rows<const M: usize, const N: usize>(
    mat: &mut Matrix<M, N>, c: f64, s: f64, r1: usize, r2: usize,
) {
    for j in 0..N {
        let a = mat[(r1, j)];
        let b = mat[(r2, j)];
        mat[(r1, j)] = c * a + s * b;
        mat[(r2, j)] = -s * a + c * b;
    }
}

/// Apply Givens rotation to two columns (c1, c2) of a matrix.
/// new_c1 = c * col_c1 + s * col_c2
/// new_c2 = -s * col_c1 + c * col_c2
fn rotate_cols<const M: usize, const N: usize>(
    mat: &mut Matrix<M, N>, c: f64, s: f64, c1: usize, c2: usize,
) {
    for i in 0..M {
        let a = mat[(i, c1)];
        let b = mat[(i, c2)];
        mat[(i, c1)] = c * a + s * b;
        mat[(i, c2)] = -s * a + c * b;
    }
}

// ============================================================
// Householder bidiagonalization
// ============================================================

/// Generate Householder vector v (v[0]=1) and beta.
/// H = I - beta * v * v^T reflects x to +/- ||x|| * e_1.
fn householder_vec(x: &[f64]) -> (Vec<f64>, f64) {
    let n = x.len();
    let mut v = x.to_vec();

    let norm_x: f64 = x.iter().map(|&xi| xi * xi).sum::<f64>().sqrt();
    if norm_x == 0.0 {
        return (vec![0.0; n], 0.0);
    }

    let alpha = if x[0] >= 0.0 { norm_x } else { -norm_x };
    v[0] += alpha;

    let scale = v[0];
    if scale.abs() < 1e-15 {
        return (vec![0.0; n], 0.0);
    }
    for i in 1..n {
        v[i] /= scale;
    }
    v[0] = 1.0;

    let vtv: f64 = 1.0 + v[1..].iter().map(|&vi| vi * vi).sum::<f64>();
    (v, 2.0 / vtv)
}

/// Householder bidiagonalization: A = U * B * V^T
/// U is M x M, B is upper bidiagonal (stored in N x N), V is N x N.
/// Requires M >= N.
fn bidiagonalize<const M: usize, const N: usize>(
    a: &Matrix<M, N>,
) -> (Matrix<M, M>, Matrix<N, N>, Matrix<N, N>) {
    // Work copy as raw 2D array
    let mut w = [[0.0f64; N]; M];
    for i in 0..M {
        for j in 0..N {
            w[i][j] = a[(i, j)];
        }
    }

    let mut u = Matrix::<M, M>::identity();
    let mut v = Matrix::<N, N>::identity();

    for k in 0..N {
        // -- Left Householder: zero column k below diagonal --
        let len = M - k;
        let x: Vec<f64> = (0..len).map(|i| w[k + i][k]).collect();
        let (hv, beta) = householder_vec(&x);

        if beta > 0.0 {
            // Apply H to work columns k..N (from left)
            for j in k..N {
                let dot: f64 = (0..len).map(|i| hv[i] * w[k + i][j]).sum();
                for i in 0..len {
                    w[k + i][j] -= beta * hv[i] * dot;
                }
            }
            // Accumulate U = U * H (H acts on indices k..M-1)
            for row in 0..M {
                let dot: f64 = (0..len).map(|i| u[(row, k + i)] * hv[i]).sum();
                for i in 0..len {
                    u[(row, k + i)] -= beta * hv[i] * dot;
                }
            }
        }

        // -- Right Householder: zero row k beyond superdiagonal --
        if k + 2 <= N {
            let len = N - k - 1;
            let x: Vec<f64> = (0..len).map(|j| w[k][k + 1 + j]).collect();
            let (hv, beta) = householder_vec(&x);

            if beta > 0.0 {
                // Apply H to work rows k..M (from right)
                for row in k..M {
                    let dot: f64 = (0..len).map(|j| w[row][k + 1 + j] * hv[j]).sum();
                    for j in 0..len {
                        w[row][k + 1 + j] -= beta * hv[j] * dot;
                    }
                }
                // Accumulate V = V * G (G acts on indices k+1..N-1)
                for row in 0..N {
                    let dot: f64 = (0..len).map(|j| v[(row, k + 1 + j)] * hv[j]).sum();
                    for j in 0..len {
                        v[(row, k + 1 + j)] -= beta * hv[j] * dot;
                    }
                }
            }
        }
    }

    // Extract upper bidiagonal into N x N matrix
    let mut b = Matrix::<N, N>::zeros();
    for i in 0..N {
        b[(i, i)] = w[i][i];
        if i + 1 < N {
            b[(i, i + 1)] = w[i][i + 1];
        }
    }

    (u, b, v)
}

// ============================================================
// Golub-Kahan QR step
// ============================================================

/// Wilkinson shift from trailing 2x2 of B^T * B.
fn wilkinson_shift<const N: usize>(b: &Matrix<N, N>, p: usize, q: usize) -> f64 {
    let d_qm1 = b[(q - 1, q - 1)];
    let d_q = b[(q, q)];
    let e_qm1 = b[(q - 1, q)];
    let e_qm2 = if q >= 2 && q - 2 >= p { b[(q - 2, q - 1)] } else { 0.0 };

    // Trailing 2x2 of T = B^T * B
    let t11 = d_qm1 * d_qm1 + e_qm2 * e_qm2;
    let t22 = d_q * d_q + e_qm1 * e_qm1;
    let t12 = d_qm1 * e_qm1;

    let delta = (t11 - t22) / 2.0;
    if delta.abs() < 1e-15 && t12.abs() < 1e-15 {
        return t22;
    }

    let sign_d = if delta >= 0.0 { 1.0 } else { -1.0 };
    t22 - t12 * t12 / (delta + sign_d * (delta * delta + t12 * t12).sqrt())
}

/// One implicit QR step (Golub-Kahan) on bidiagonal B[p..=q].
/// Accumulates Givens rotations into U (M x M) and V (N x N).
fn gk_step<const M: usize, const N: usize>(
    b: &mut Matrix<N, N>,
    u: &mut Matrix<M, M>,
    v: &mut Matrix<N, N>,
    p: usize,
    q: usize,
) {
    let mu = wilkinson_shift(b, p, q);

    // Initial bulge from first column of B^T*B - mu*I
    let mut y = b[(p, p)] * b[(p, p)] - mu;
    let mut z = b[(p, p)] * b[(p, p + 1)];

    for k in p..q {
        // Right Givens: rotate columns k, k+1 to zero z
        let (c, s) = givens(y, z);
        rotate_cols(b, c, s, k, k + 1);
        rotate_cols(v, c, s, k, k + 1);

        // Now bulge at (k+1, k)
        y = b[(k, k)];
        z = b[(k + 1, k)];

        // Left Givens: rotate rows k, k+1 to zero b[(k+1, k)]
        let (c, s) = givens(y, z);
        rotate_rows(b, c, s, k, k + 1);
        rotate_cols(u, c, s, k, k + 1); // U = U * G, so column rotation

        // Setup next bulge
        if k < q - 1 {
            y = b[(k, k + 1)];
            z = b[(k, k + 2)];
        }
    }
}

// ============================================================
// Main SVD driver
// ============================================================

/// SVD via Golub-Kahan bidiagonalization + implicit QR shifts.
/// Requires M >= N. For fat matrices, transpose first.
pub fn svd_golub_kahan<const M: usize, const N: usize>(
    a: &Matrix<M, N>,
) -> SVD<M, N, N> {
    assert!(M >= N, "svd_golub_kahan requires M >= N");
    assert!(N >= 2, "Need at least 2 columns");

    // Step 1: Bidiagonalize
    let (mut u, mut b, mut v) = bidiagonalize(a);

    // Step 2: Iterative QR on bidiagonal
    let max_total = MAX_ITER * N;
    for _ in 0..max_total {
        // Zero negligible superdiagonal entries
        for i in 0..N - 1 {
            if b[(i, i + 1)].abs() <= EPS * (b[(i, i)].abs() + b[(i + 1, i + 1)].abs()) {
                b[(i, i + 1)] = 0.0;
            }
        }

        // Find q: largest index such that B[q-1, q] != 0
        let mut q = 0;
        for i in (0..N - 1).rev() {
            if b[(i, i + 1)].abs() > 0.0 {
                q = i + 1;
                break;
            }
        }

        if q == 0 {
            break; // B is diagonal, converged
        }

        // Find p: start of unreduced block ending at q
        let mut p = 0;
        for i in (0..q).rev() {
            if b[(i, i + 1)].abs() == 0.0 {
                p = i + 1;
                break;
            }
        }

        // Check for zero diagonal entries in [p, q]
        let mut handled_zero = false;
        for i in p..=q {
            if b[(i, i)].abs() < EPS {
                handled_zero = true;
                if i < q {
                    // d_i = 0: chase b[(i, i+1)] downward using left Givens
                    // Rotate rows i and i+1, then i and i+2, etc.
                    let mut bulge_col = i + 1;
                    for j in (i + 1)..=q {
                        if b[(i, bulge_col)].abs() < 1e-15 {
                            break;
                        }
                        let (c, s) = givens(b[(j, bulge_col)], b[(i, bulge_col)]);
                        rotate_rows(&mut b, c, s, j, i);
                        rotate_cols(&mut u, c, s, j, i);
                        // After rotation, b[(i, bulge_col)] = 0
                        // but fill-in may appear at b[(i, bulge_col+1)]
                        if bulge_col + 1 <= q {
                            bulge_col += 1;
                        } else {
                            break;
                        }
                    }
                } else {
                    // d_q = 0: chase b[(q-1, q)] upward using left Givens
                    let mut bulge_row = q - 1;
                    for j in (p..q).rev() {
                        if b[(bulge_row, q)].abs() < 1e-15 {
                            break;
                        }
                        let (c, s) = givens(b[(j, j)], b[(j, q)]);
                        // Rotate columns j and q
                        rotate_cols(&mut b, c, s, j, q);
                        rotate_cols(&mut v, c, s, j, q);
                        // fill-in may appear at b[(j-1, q)] if j > p
                        if j > p {
                            bulge_row = j - 1;
                        } else {
                            break;
                        }
                    }
                }
                break;
            }
        }

        if !handled_zero && p < q {
            gk_step(&mut b, &mut u, &mut v, p, q);
        }
    }

    // Step 3: Extract singular values, ensure positive
    let mut s = [0.0f64; N];
    for i in 0..N {
        s[i] = b[(i, i)].abs();
        if b[(i, i)] < 0.0 {
            // Flip sign of corresponding U column
            for j in 0..M {
                u[(j, i)] = -u[(j, i)];
            }
        }
    }

    // Step 4: Sort descending
    for i in 0..N {
        let mut max_idx = i;
        for j in (i + 1)..N {
            if s[j] > s[max_idx] {
                max_idx = j;
            }
        }
        if max_idx != i {
            s.swap(i, max_idx);
            for r in 0..M {
                let tmp = u[(r, i)];
                u[(r, i)] = u[(r, max_idx)];
                u[(r, max_idx)] = tmp;
            }
            for r in 0..N {
                let tmp = v[(r, i)];
                v[(r, i)] = v[(r, max_idx)];
                v[(r, max_idx)] = tmp;
            }
        }
    }

    // Step 5: Extract thin U (M x N from M x M)
    let mut u_thin = Matrix::<M, N>::zeros();
    for i in 0..M {
        for j in 0..N {
            u_thin[(i, j)] = u[(i, j)];
        }
    }

    SVD { u: u_thin, s, v }
}
