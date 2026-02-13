use lalir::matrix::*;
use lalir::cholesky::*;

const TOL: f64 = 1e-10;

// ============================================================
// Cholesky (LL?) tests
// ============================================================

#[test]
fn test_cholesky_solve() {
    // Same SPD matrix and RHS as C test
    let a = Matrix::<3, 3>::from_slice(&[
        4.0, 2.0, 2.0,
        2.0, 10.0, 4.0,
        2.0, 4.0, 9.0,
    ]);
    let b = Vector::<3>::from_array([1.0, 2.0, 3.0]);

    let chol = cholesky_decompose(&a).expect("Cholesky failed");
    let x = chol.solve(&b);

    // Verify A * x \approx b
    let residual = (a.matvec(&x) - b).norm();
    assert!(residual < TOL, "Cholesky residual too large: {}", residual);
}

#[test]
fn test_cholesky_l_is_lower_triangular() {
    let a = Matrix::<3, 3>::from_slice(&[
        4.0, 2.0, 2.0,
        2.0, 10.0, 4.0,
        2.0, 4.0, 9.0,
    ]);

    let chol = cholesky_decompose(&a).expect("Cholesky failed");

    // Upper triangle should be zero
    for i in 0..3 {
        for j in (i + 1)..3 {
            assert!(
                chol.l[(i, j)].abs() < TOL,
                "L not lower triangular at ({},{}): {}", i, j, chol.l[(i, j)]
            );
        }
    }
}

#[test]
fn test_cholesky_reconstruction() {
    // Verify L * L? = A
    let a = Matrix::<3, 3>::from_slice(&[
        4.0, 2.0, 2.0,
        2.0, 10.0, 4.0,
        2.0, 4.0, 9.0,
    ]);

    let chol = cholesky_decompose(&a).expect("Cholesky failed");
    let lt = chol.l.transpose();
    let recon = chol.l * lt;

    let err = (recon - a).norm_fro();
    assert!(err < TOL, "L*L? != A, error: {}", err);
}

#[test]
fn test_cholesky_identity() {
    let a = Matrix::<3, 3>::identity();
    let chol = cholesky_decompose(&a).expect("Cholesky of I failed");

    // L should be identity
    let err = (chol.l - Matrix::<3, 3>::identity()).norm_fro();
    assert!(err < TOL, "Cholesky of I should give I");

    // Solve Ix = b => x = b
    let b = Vector::<3>::from_array([7.0, -3.0, 42.0]);
    let x = chol.solve(&b);
    let residual = (x - b).norm();
    assert!(residual < TOL, "Identity solve failed");
}

#[test]
fn test_cholesky_not_spd() {
    // Not positive definite (has negative eigenvalue)
    let a = Matrix::<3, 3>::from_slice(&[
         1.0, 2.0, 3.0,
         2.0, 1.0, 4.0,
         3.0, 4.0, 1.0,
    ]);
    assert!(cholesky_decompose(&a).is_err(), "Should reject non-SPD matrix");
}

#[test]
fn test_cholesky_4x4() {
    // Larger SPD: A = B * B? + I ensures SPD
    let b = Matrix::<4, 4>::from_slice(&[
        2.0, 1.0, 0.0, 1.0,
        1.0, 3.0, 1.0, 0.0,
        0.0, 1.0, 2.0, 1.0,
        1.0, 0.0, 1.0, 4.0,
    ]);
    let a = b * b.transpose() + Matrix::<4, 4>::identity();

    let rhs = Vector::<4>::from_array([1.0, 2.0, 3.0, 4.0]);
    let chol = cholesky_decompose(&a).expect("Cholesky 4x4 failed");
    let x = chol.solve(&rhs);

    let residual = (a.matvec(&x) - rhs).norm();
    assert!(residual < TOL, "4x4 Cholesky residual: {}", residual);
}

// ============================================================
// LDL? tests
// ============================================================

#[test]
fn test_ldlt_solve() {
    // Same matrix/RHS as C test
    let a = Matrix::<3, 3>::from_slice(&[
        4.0, 2.0, 2.0,
        2.0, 10.0, 4.0,
        2.0, 4.0, 9.0,
    ]);
    let b = Vector::<3>::from_array([1.0, 2.0, 3.0]);

    let ldlt = ldlt_decompose(&a).expect("LDLT failed");
    let x = ldlt.solve(&b);

    let residual = (a.matvec(&x) - b).norm();
    assert!(residual < TOL, "LDLT residual too large: {}", residual);
}

#[test]
fn test_ldlt_reconstruction() {
    // Verify L * D * L? = A
    let a = Matrix::<3, 3>::from_slice(&[
        4.0, 2.0, 2.0,
        2.0, 10.0, 4.0,
        2.0, 4.0, 9.0,
    ]);

    let ldlt = ldlt_decompose(&a).expect("LDLT failed");

    // Build D as a matrix
    let mut d_mat = Matrix::<3, 3>::zeros();
    for i in 0..3 {
        d_mat[(i, i)] = ldlt.d[i];
    }

    let lt = ldlt.l.transpose();
    let recon = ldlt.l * d_mat * lt;

    let err = (recon - a).norm_fro();
    assert!(err < TOL, "L*D*L? != A, error: {}", err);
}

#[test]
fn test_ldlt_unit_lower_triangular() {
    let a = Matrix::<3, 3>::from_slice(&[
        4.0, 2.0, 2.0,
        2.0, 10.0, 4.0,
        2.0, 4.0, 9.0,
    ]);

    let ldlt = ldlt_decompose(&a).expect("LDLT failed");

    // Diagonal should be 1s
    for i in 0..3 {
        assert!(
            (ldlt.l[(i, i)] - 1.0).abs() < TOL,
            "L diagonal not 1 at {}: {}", i, ldlt.l[(i, i)]
        );
    }

    // Upper triangle should be zero
    for i in 0..3 {
        for j in (i + 1)..3 {
            assert!(
                ldlt.l[(i, j)].abs() < TOL,
                "L not lower triangular at ({},{}): {}", i, j, ldlt.l[(i, j)]
            );
        }
    }
}

#[test]
fn test_ldlt_agrees_with_cholesky() {
    // Both should give the same solution for SPD matrices
    let a = Matrix::<3, 3>::from_slice(&[
        4.0, 2.0, 2.0,
        2.0, 10.0, 4.0,
        2.0, 4.0, 9.0,
    ]);
    let b = Vector::<3>::from_array([1.0, 2.0, 3.0]);

    let x_chol = cholesky_decompose(&a).unwrap().solve(&b);
    let x_ldlt = ldlt_decompose(&a).unwrap().solve(&b);

    let diff = (x_chol - x_ldlt).norm();
    assert!(diff < TOL, "Cholesky and LDLT disagree: {}", diff);
}

#[test]
fn test_ldlt_singular() {
    let a = Matrix::<3, 3>::from_slice(&[
        1.0, 2.0, 3.0,
        2.0, 4.0, 6.0,
        3.0, 6.0, 9.0,
    ]);
    assert!(ldlt_decompose(&a).is_err(), "Should detect singular matrix");
}

#[test]
fn test_ldlt_4x4() {
    let b = Matrix::<4, 4>::from_slice(&[
        2.0, 1.0, 0.0, 1.0,
        1.0, 3.0, 1.0, 0.0,
        0.0, 1.0, 2.0, 1.0,
        1.0, 0.0, 1.0, 4.0,
    ]);
    let a = b * b.transpose() + Matrix::<4, 4>::identity();

    let rhs = Vector::<4>::from_array([1.0, 2.0, 3.0, 4.0]);
    let ldlt = ldlt_decompose(&a).expect("LDLT 4x4 failed");
    let x = ldlt.solve(&rhs);

    let residual = (a.matvec(&x) - rhs).norm();
    assert!(residual < TOL, "4x4 LDLT residual: {}", residual);
}
