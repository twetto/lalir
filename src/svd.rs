use crate::matrix::Matrix;
use crate::eigen::{eigen_symmetric_qr, eigen_symmetric_jacobi, sort_eigenpairs};

/// SVD result: A = U * diag(S) * V^T
/// U is MxR, S has R singular values, V is NxR, where R = min(M, N).
pub struct SVD<const M: usize, const N: usize, const R: usize> {
    pub u: Matrix<M, R>,
    pub s: [f64; R],
    pub v: Matrix<N, R>,
}

/// SVD via eigen-decomposition of A^T A using QR iteration.
/// For M >= N (tall or square). R = N.
pub fn svd_qr<const M: usize, const N: usize>(
    a: &Matrix<M, N>,
) -> SVD<M, N, N> {
    // B = A^T A (NxN)
    let at = a.transpose();
    let b = at * *a;

    // Eigen-decompose B = V \Lambda V^T
    let (mut eigenvalues, mut eigvecs) = eigen_symmetric_qr(&b);
    sort_eigenpairs(&mut eigenvalues, &mut eigvecs);

    // Singular values = sqrt(eigenvalues)
    let mut s = [0.0f64; N];
    for i in 0..N {
        s[i] = eigenvalues[i].max(0.0).sqrt();
    }

    // U = A * V * \Sigma^{-1}
    let av = *a * eigvecs;
    let mut u = Matrix::<M, N>::zeros();
    for i in 0..N {
        if s[i] > 1e-12 {
            for j in 0..M {
                u[(j, i)] = av[(j, i)] / s[i];
            }
        }
    }

    // V = eigvecs
    SVD { u, s, v: eigvecs }
}

/// SVD via eigen-decomposition of the augmented symmetric matrix
/// M = [0 A; A^T 0] using Jacobi iteration.
/// For M >= N (tall or square). R = N.
///
/// Note: uses a fixed-size augmented matrix. We need M+N as a const generic,
/// but Rust doesn't support const expressions in generics yet.
/// So we implement this for specific small sizes via a helper.
pub fn svd_augmented<const M: usize, const N: usize, const MN: usize>(
    a: &Matrix<M, N>,
) -> SVD<M, N, N> {
    assert!(MN == M + N, "MN must equal M + N");

    // Build augmented matrix [0 A; A^T 0]
    let mut aug = Matrix::<MN, MN>::zeros();
    for i in 0..M {
        for j in 0..N {
            aug[(i, M + j)] = a[(i, j)];
            aug[(M + j, i)] = a[(i, j)];
        }
    }

    // Eigen-decompose
    let (mut eigenvals, mut eigvecs) = eigen_symmetric_jacobi(&aug);
    sort_eigenpairs(&mut eigenvals, &mut eigvecs);

    // Extract singular values and vectors from positive eigenvalues
    let mut s = [0.0f64; N];
    let mut u = Matrix::<M, N>::zeros();
    let mut v = Matrix::<N, N>::zeros();

    let mut count = 0;
    let sqrt2 = std::f64::consts::SQRT_2;
    for i in 0..MN {
        if count >= N {
            break;
        }
        let sigma = eigenvals[i].abs();
        if sigma < 1e-12 {
            continue;
        }

        s[count] = sigma;

        // U: top M rows of eigenvector i, scaled by sqrt(2)
        for j in 0..M {
            u[(j, count)] = eigvecs[(j, i)] * sqrt2;
        }
        // V: bottom N rows of eigenvector i, scaled by sqrt(2)
        for j in 0..N {
            v[(j, count)] = eigvecs[(M + j, i)] * sqrt2;
        }

        count += 1;
    }

    SVD { u, s, v }
}

/// SVD for fat matrices (M < N): compute via transpose.
/// Returns SVD<M, N, M> where R = M.
pub fn svd_fat<const M: usize, const N: usize>(
    a: &Matrix<M, N>,
) -> SVD<M, N, M> {
    let at = a.transpose();
    // SVD of A^T (which is NxM, tall): A^T = U' S V'^T
    // => A = V' S U'^T
    let svd_t = svd_qr(&at);

    SVD {
        u: svd_t.v,  // V' of A^T becomes U of A
        s: svd_t.s,
        v: svd_t.u,  // U' of A^T becomes V of A
    }
}

/// Reconstruct A from SVD factors: A = U * diag(S) * V^T
pub fn reconstruct<const M: usize, const N: usize, const R: usize>(
    svd: &SVD<M, N, R>,
) -> Matrix<M, N> {
    let mut a = Matrix::<M, N>::zeros();
    for k in 0..R {
        let sk = svd.s[k];
        for i in 0..M {
            for j in 0..N {
                a[(i, j)] += svd.u[(i, k)] * sk * svd.v[(j, k)];
            }
        }
    }
    a
}

/// Compute numerical rank from singular values.
pub fn numerical_rank<const R: usize>(s: &[f64; R], tol: f64) -> usize {
    s.iter().filter(|&&si| si > tol).count()
}
