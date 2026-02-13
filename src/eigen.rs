use crate::matrix::Matrix;
use crate::qr::qr_householder;

const MAX_ITER: usize = 1000;
const TOL: f64 = 1e-10;

/// Eigen-decompose a real symmetric matrix: A = V * \Lambda * V^T
/// Uses QR iteration. Returns eigenvalues and eigenvector matrix V.
pub fn eigen_symmetric_qr<const N: usize>(
    a: &Matrix<N, N>,
) -> ([f64; N], Matrix<N, N>) {
    let mut a_k = *a;
    let mut v = Matrix::<N, N>::identity();

    for _ in 0..MAX_ITER {
        let hqr = qr_householder(&a_k);
        let q = hqr.q();
        let r = hqr.r();

        // A_{k+1} = R * Q
        a_k = r * q;

        // V = V * Q
        v = v * q;

        // Check convergence: sum of off-diagonal elements
        let mut offdiag = 0.0;
        for i in 0..N {
            for j in 0..N {
                if i != j {
                    offdiag += a_k[(i, j)].abs();
                }
            }
        }
        if offdiag < TOL {
            break;
        }
    }

    let mut eigenvalues = [0.0f64; N];
    for i in 0..N {
        eigenvalues[i] = a_k[(i, i)];
    }

    (eigenvalues, v)
}

/// Eigen-decompose a real symmetric matrix: A = V * \Lambda * V^T
/// Uses Jacobi iteration (more stable, good for augmented SVD).
pub fn eigen_symmetric_jacobi<const N: usize>(
    a: &Matrix<N, N>,
) -> ([f64; N], Matrix<N, N>) {
    let mut a_k = *a;
    let mut v = Matrix::<N, N>::identity();

    for _ in 0..MAX_ITER {
        // Find largest off-diagonal element
        let mut max_val = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..N {
            for j in (i + 1)..N {
                let aij = a_k[(i, j)].abs();
                if aij > max_val {
                    max_val = aij;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < TOL {
            break;
        }

        let app = a_k[(p, p)];
        let aqq = a_k[(q, q)];
        let apq = a_k[(p, q)];

        let phi = 0.5 * (2.0 * apq).atan2(aqq - app);
        let c = phi.cos();
        let s = phi.sin();

        // Apply Givens rotation: A = G^T A G
        // First: A = G^T A (rotate rows p, q)
        for k in 0..N {
            let aik = a_k[(p, k)];
            let aqk = a_k[(q, k)];
            a_k[(p, k)] = c * aik - s * aqk;
            a_k[(q, k)] = s * aik + c * aqk;
        }
        // Then: A = A G (rotate columns p, q)
        for k in 0..N {
            let akp = a_k[(k, p)];
            let akq = a_k[(k, q)];
            a_k[(k, p)] = c * akp - s * akq;
            a_k[(k, q)] = s * akp + c * akq;
        }

        // Accumulate V = V * G
        for k in 0..N {
            let vip = v[(k, p)];
            let viq = v[(k, q)];
            v[(k, p)] = c * vip - s * viq;
            v[(k, q)] = s * vip + c * viq;
        }
    }

    let mut eigenvalues = [0.0f64; N];
    for i in 0..N {
        eigenvalues[i] = a_k[(i, i)];
    }

    (eigenvalues, v)
}

/// Sort eigenvalues (and corresponding eigenvectors) in descending order.
pub fn sort_eigenpairs<const N: usize>(
    eigenvalues: &mut [f64; N],
    eigvecs: &mut Matrix<N, N>,
) {
    for i in 0..N {
        let mut max_idx = i;
        for j in (i + 1)..N {
            if eigenvalues[j] > eigenvalues[max_idx] {
                max_idx = j;
            }
        }
        if max_idx != i {
            eigenvalues.swap(i, max_idx);
            // Swap columns i and max_idx
            for row in 0..N {
                let tmp = eigvecs[(row, i)];
                eigvecs[(row, i)] = eigvecs[(row, max_idx)];
                eigvecs[(row, max_idx)] = tmp;
            }
        }
    }
}
