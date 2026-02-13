use lalir::matrix::*;
use lalir::qr::*;

const TOL: f64 = 1e-10;

fn assert_mat_near<const M: usize, const N: usize>(
    a: &Matrix<M, N>, b: &Matrix<M, N>, tol: f64, label: &str,
) {
    let err = (*a - *b).norm_fro();
    assert!(err < tol, "{}: error {} exceeds tol {}", label, err, tol);
}

fn assert_orthogonal<const N: usize>(q: &Matrix<N, N>, label: &str) {
    let qtq = q.transpose() * *q;
    assert_mat_near(&qtq, &Matrix::<N, N>::identity(), TOL, &format!("{}?*{} = I", label, label));
}

// ============================================================
// Gram-Schmidt tests (matching test_qr_gram_schmidt)
// ============================================================

#[test]
fn test_gram_schmidt_orthogonality() {
    let a = Matrix::<3, 3>::from_slice(&[
        12.0, -51.0,   4.0,
         6.0, 167.0, -68.0,
        -4.0,  24.0, -41.0,
    ]);

    let qr = qr_gram_schmidt(&a).expect("Gram-Schmidt failed");
    assert_orthogonal(&qr.q, "Q");
}

#[test]
fn test_gram_schmidt_reconstruction() {
    let a = Matrix::<3, 3>::from_slice(&[
        12.0, -51.0,   4.0,
         6.0, 167.0, -68.0,
        -4.0,  24.0, -41.0,
    ]);

    let qr = qr_gram_schmidt(&a).expect("Gram-Schmidt failed");
    let recon = qr.q * qr.r;
    assert_mat_near(&recon, &a, 1e-8, "Q*R = A");
}

#[test]
fn test_gram_schmidt_solve() {
    let a = Matrix::<3, 3>::from_slice(&[
        12.0, -51.0,   4.0,
         6.0, 167.0, -68.0,
        -4.0,  24.0, -41.0,
    ]);
    let b = Vector::<3>::from_array([1.0, 2.0, 3.0]);

    let qr = qr_gram_schmidt(&a).expect("Gram-Schmidt failed");

    // Solve via Q?b then back-substitute on R
    let qt_b = qr.q.transpose().matvec(&b);
    // Extract top 3x3 of R for back-sub
    let mut x = Vector::<3>::zeros();
    for i in (0..3).rev() {
        let mut sum = qt_b[i];
        for j in (i + 1)..3 {
            sum -= qr.r[(i, j)] * x[j];
        }
        x[i] = sum / qr.r[(i, i)];
    }

    let residual = (a.matvec(&x) - b).norm();
    assert!(residual < TOL, "Gram-Schmidt solve residual: {}", residual);
}

// ============================================================
// Householder tests (matching test_qr_full_rank)
// ============================================================

#[test]
fn test_householder_q_orthogonal() {
    let a = Matrix::<3, 3>::from_slice(&[
        12.0, -51.0,   4.0,
         6.0, 167.0, -68.0,
        -4.0,  24.0, -41.0,
    ]);

    let hqr = qr_householder(&a);
    let q = hqr.q();
    assert_orthogonal(&q, "Q");
}

#[test]
fn test_householder_reconstruction() {
    let a = Matrix::<3, 3>::from_slice(&[
        12.0, -51.0,   4.0,
         6.0, 167.0, -68.0,
        -4.0,  24.0, -41.0,
    ]);

    let hqr = qr_householder(&a);
    let q = hqr.q();
    let r = hqr.r();
    let recon = q * r;
    assert_mat_near(&recon, &a, 1e-8, "Q*R = A (Householder)");
}

#[test]
fn test_householder_solve() {
    let a = Matrix::<3, 3>::from_slice(&[
        12.0, -51.0,   4.0,
         6.0, 167.0, -68.0,
        -4.0,  24.0, -41.0,
    ]);
    let b = Vector::<3>::from_array([1.0, 2.0, 3.0]);

    let hqr = qr_householder(&a);
    let x = hqr.solve(&b);

    let residual = (a.matvec(&x) - b).norm();
    assert!(residual < TOL, "Householder solve residual: {}", residual);
}

#[test]
fn test_householder_full_rank() {
    let a = Matrix::<3, 3>::from_slice(&[
        12.0, -51.0,   4.0,
         6.0, 167.0, -68.0,
        -4.0,  24.0, -41.0,
    ]);

    let hqr = qr_householder(&a);
    let tol = hqr.packed.max_abs_diag() * 1e-6 * 3.0;
    assert!(hqr.is_full_rank(tol), "Should be full rank");
}

// ============================================================
// Rank-deficient test (matching test_qr_rank_deficient)
// ============================================================

#[test]
fn test_householder_rank_deficient() {
    let a = Matrix::<3, 3>::from_slice(&[
        1.0, 2.0, 3.0,
        5.0, 7.0, 9.0,
        4.0, 5.0, 6.0,
    ]);

    let hqr = qr_householder(&a);
    let tol = hqr.packed.max_abs_diag() * 1e-6 * 3.0;
    assert!(!hqr.is_full_rank(tol), "Should detect rank deficiency");
}

#[test]
fn test_gram_schmidt_rejects_rank_deficient() {
    let a = Matrix::<3, 3>::from_slice(&[
        1.0, 2.0, 3.0,
        5.0, 7.0, 9.0,
        4.0, 5.0, 6.0,
    ]);

    assert!(qr_gram_schmidt(&a).is_err(), "Gram-Schmidt should fail on rank-deficient");
}

// ============================================================
// Givens tests
// ============================================================

#[test]
fn test_givens_orthogonality() {
    let a = Matrix::<3, 3>::from_slice(&[
        12.0, -51.0,   4.0,
         6.0, 167.0, -68.0,
        -4.0,  24.0, -41.0,
    ]);

    let qr = qr_givens(&a);
    assert_orthogonal(&qr.q, "Q (Givens)");
}

#[test]
fn test_givens_reconstruction() {
    let a = Matrix::<3, 3>::from_slice(&[
        12.0, -51.0,   4.0,
         6.0, 167.0, -68.0,
        -4.0,  24.0, -41.0,
    ]);

    let qr = qr_givens(&a);
    let recon = qr.q * qr.r;
    assert_mat_near(&recon, &a, 1e-8, "Q*R = A (Givens)");
}

// ============================================================
// Least-squares tests (matching test_least_squares_qr)
// ============================================================

#[test]
fn test_least_squares_householder() {
    // Overdetermined: 3x2
    let a = Matrix::<3, 2>::from_slice(&[
        1.0, 2.0,
        2.0, 1.0,
        3.0, 4.0,
    ]);
    let b = Vector::<3>::from_array([4.0, 3.0, 10.0]);

    let hqr = qr_householder(&a);
    let x = hqr.solve_least_squares(&b);

    // Verify normal equations: A^T(Ax - b) \approx 0
    let residual = a.matvec(&x) - b;
    let at = a.transpose();
    let normal_residual = at.matvec(&residual);
    assert!(
        normal_residual.norm() < 1e-8,
        "Normal equations residual: {}", normal_residual.norm()
    );
}

#[test]
fn test_least_squares_qrcp() {
    let a = Matrix::<3, 2>::from_slice(&[
        1.0, 2.0,
        2.0, 1.0,
        3.0, 4.0,
    ]);
    let b = Vector::<3>::from_array([4.0, 3.0, 10.0]);

    let pqr = qr_col_pivoted(&a);
    let x_cp = pqr.solve_least_squares(&b);

    // Compare with Householder (should agree)
    let hqr = qr_householder(&a);
    let x_qr = hqr.solve_least_squares(&b);

    let diff = (x_cp - x_qr).norm();
    assert!(diff < 1e-8, "QRCP and QR disagree: {}", diff);
}

#[test]
fn test_least_squares_check_ax() {
    let a = Matrix::<3, 2>::from_slice(&[
        1.0, 2.0,
        2.0, 1.0,
        3.0, 4.0,
    ]);
    let b = Vector::<3>::from_array([4.0, 3.0, 10.0]);

    let hqr = qr_householder(&a);
    let x = hqr.solve_least_squares(&b);
    let ax = a.matvec(&x);

    // Ax should be the projection of b onto col(A)
    // Verify A^T(Ax - b) = 0 (optimality condition)
    let at = a.transpose();
    let opt = at.matvec(&(ax - b));
    assert!(opt.norm() < 1e-8, "Optimality condition violated: {}", opt.norm());
}

// ============================================================
// All three methods agree
// ============================================================

#[test]
fn test_all_methods_agree() {
    let a = Matrix::<3, 3>::from_slice(&[
        12.0, -51.0,   4.0,
         6.0, 167.0, -68.0,
        -4.0,  24.0, -41.0,
    ]);
    let b = Vector::<3>::from_array([1.0, 2.0, 3.0]);

    // Householder
    let x_hh = qr_householder(&a).solve(&b);

    // Gram-Schmidt (manual solve)
    let gs = qr_gram_schmidt(&a).unwrap();
    let qt_b = gs.q.transpose().matvec(&b);
    let mut x_gs = Vector::<3>::zeros();
    for i in (0..3).rev() {
        let mut sum = qt_b[i];
        for j in (i + 1)..3 {
            sum -= gs.r[(i, j)] * x_gs[j];
        }
        x_gs[i] = sum / gs.r[(i, i)];
    }

    let diff = (x_hh - x_gs).norm();
    assert!(diff < 1e-8, "Householder vs Gram-Schmidt: {}", diff);
}

// ============================================================
// Rectangular (tall) Householder
// ============================================================

#[test]
fn test_householder_tall_reconstruction() {
    let a = Matrix::<4, 3>::from_slice(&[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 10.0,
        1.0, 0.0, 1.0,
    ]);

    let hqr = qr_householder(&a);
    let q = hqr.q();
    let r = hqr.r();

    assert_orthogonal(&q, "Q (4x4)");

    let recon = q * r;
    assert_mat_near(&recon, &a, 1e-8, "Q*R = A (tall)");
}
