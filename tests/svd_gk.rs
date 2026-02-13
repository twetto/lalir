use lalir::matrix::*;
use lalir::svd::{reconstruct, numerical_rank};
use lalir::svd_gk::svd_golub_kahan;

const TOL: f64 = 1e-6;

// ============================================================
// Basic 3x2 tall matrix
// ============================================================

#[test]
fn test_gk_3x2() {
    let a = Matrix::<3, 2>::from_slice(&[
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    ]);

    let svd = svd_golub_kahan(&a);

    assert!(svd.s[0] >= svd.s[1], "Singular values not sorted");

    let recon = reconstruct(&svd);
    let err = (recon - a).norm_fro();
    assert!(err < TOL, "GK reconstruction error: {}", err);
}

#[test]
fn test_gk_singular_values_known() {
    // sigma_1 ~ 9.5255, sigma_2 ~ 0.5143
    let a = Matrix::<3, 2>::from_slice(&[
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    ]);

    let svd = svd_golub_kahan(&a);
    assert!((svd.s[0] - 9.5255).abs() < 0.001, "s0 = {}", svd.s[0]);
    assert!((svd.s[1] - 0.5143).abs() < 0.001, "s1 = {}", svd.s[1]);
}

// ============================================================
// Orthogonality of U and V
// ============================================================

#[test]
fn test_gk_u_orthonormal() {
    let a = Matrix::<3, 2>::from_slice(&[
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    ]);

    let svd = svd_golub_kahan(&a);
    let utu = svd.u.transpose() * svd.u;
    let err = (utu - Matrix::<2, 2>::identity()).norm_fro();
    assert!(err < TOL, "U^T U not identity: {}", err);
}

#[test]
fn test_gk_v_orthogonal() {
    let a = Matrix::<3, 2>::from_slice(&[
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    ]);

    let svd = svd_golub_kahan(&a);
    let vtv = svd.v.transpose() * svd.v;
    let err = (vtv - Matrix::<2, 2>::identity()).norm_fro();
    assert!(err < TOL, "V^T V not identity: {}", err);
}

// ============================================================
// Square matrix
// ============================================================

#[test]
fn test_gk_square_3x3() {
    let a = Matrix::<3, 3>::from_slice(&[
        12.0, -51.0,   4.0,
         6.0, 167.0, -68.0,
        -4.0,  24.0, -41.0,
    ]);

    let svd = svd_golub_kahan(&a);

    let recon = reconstruct(&svd);
    let err = (recon - a).norm_fro();
    assert!(err < TOL, "3x3 GK reconstruction error: {}", err);
}

// ============================================================
// Identity matrix
// ============================================================

#[test]
fn test_gk_identity() {
    let a = Matrix::<3, 3>::identity();
    let svd = svd_golub_kahan(&a);

    for i in 0..3 {
        assert!(
            (svd.s[i] - 1.0).abs() < TOL,
            "Identity s_{} = {}", i, svd.s[i]
        );
    }

    let recon = reconstruct(&svd);
    let err = (recon - a).norm_fro();
    assert!(err < TOL, "Identity GK reconstruction error: {}", err);
}

// ============================================================
// Rank-deficient
// ============================================================

#[test]
fn test_gk_rank_deficient() {
    // Row 3 = Row 1 + Row 2 => rank 2
    let a = Matrix::<3, 3>::from_slice(&[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        5.0, 7.0, 9.0,
    ]);

    let svd = svd_golub_kahan(&a);

    let recon = reconstruct(&svd);
    let err = (recon - a).norm_fro();
    assert!(err < TOL, "Rank-deficient GK reconstruction error: {}", err);

    let rank = numerical_rank(&svd.s, 1e-6);
    assert_eq!(rank, 2, "Expected rank 2, got {}", rank);
}

// ============================================================
// 4x3 tall matrix
// ============================================================

#[test]
fn test_gk_4x3() {
    let a = Matrix::<4, 3>::from_slice(&[
         1.0,  0.0,  1.0,
         0.0,  1.0,  1.0,
         1.0,  1.0,  0.0,
         1.0,  1.0,  1.0,
    ]);

    let svd = svd_golub_kahan(&a);

    let recon = reconstruct(&svd);
    let err = (recon - a).norm_fro();
    assert!(err < TOL, "4x3 GK reconstruction error: {}", err);
}

// ============================================================
// Larger matrix (5x5)
// ============================================================

#[test]
fn test_gk_5x5() {
    let a = Matrix::<5, 5>::from_slice(&[
         2.0,  5.0, -10.0, -7.0, -7.0,
        -4.0,  2.0,  -9.0, -4.0, -3.0,
        -1.0, 10.0,   9.0,  6.0,  9.0,
        -7.0,  7.0,   9.0,  9.0,  9.0,
       -10.0,  0.0,  10.0, -7.0,  1.0,
    ]);

    let svd = svd_golub_kahan(&a);

    for i in 0..5 {
        assert!(svd.s[i] > 1e-6, "s_{} too small: {}", i, svd.s[i]);
    }

    let recon = reconstruct(&svd);
    let err = (recon - a).norm_fro();
    assert!(err < TOL, "5x5 GK reconstruction error: {}", err);
}

// ============================================================
// Agreement with ATA-based SVD
// ============================================================

#[test]
fn test_gk_agrees_with_qr_svd() {
    use lalir::svd::svd_qr;

    let a = Matrix::<3, 2>::from_slice(&[
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    ]);

    let svd1 = svd_qr(&a);
    let svd2 = svd_golub_kahan(&a);

    for i in 0..2 {
        assert!(
            (svd1.s[i] - svd2.s[i]).abs() < 0.001,
            "Singular values disagree at {}: {} vs {}", i, svd1.s[i], svd2.s[i]
        );
    }

    let recon1 = reconstruct(&svd1);
    let recon2 = reconstruct(&svd2);
    let err = (recon1 - recon2).norm_fro();
    assert!(err < TOL, "Reconstructions differ: {}", err);
}

// ============================================================
// 10x10
// ============================================================

#[test]
fn test_gk_10x10() {
    let a = Matrix::<10, 10>::from_slice(&[
         2.0,  5.0, -10.0,  -7.0,  -7.0,  -3.0,  -1.0,   9.0,   8.0,  -6.0,
        -4.0,  2.0,  -9.0,  -4.0,  -3.0,   4.0,   7.0,  -5.0,   3.0,  -2.0,
        -1.0, 10.0,   9.0,   6.0,   9.0,  -5.0,   5.0,   5.0, -10.0,   8.0,
        -7.0,  7.0,   9.0,   9.0,   9.0,   4.0,  -3.0, -10.0,  -9.0,  -1.0,
       -10.0,  0.0,  10.0,  -7.0,   1.0,   8.0,  -8.0, -10.0, -10.0,  -6.0,
        -5.0, -4.0,  -2.0,  10.0,   7.0,   5.0,  -6.0,  -1.0,   0.0,  -9.0,
        -9.0, -3.0,  -1.0,  -7.0,  -4.0,   1.0,   4.0,   8.0, -10.0,   4.0,
        -7.0,  2.0,   0.0,  10.0,   1.0,  -6.0,  -4.0,  -6.0,   5.0,  10.0,
        -7.0,  2.0,  -6.0,  10.0,  -2.0,   4.0,   5.0,  10.0,  -7.0,   5.0,
         3.0,  6.0,   7.0,  -5.0,  -1.0,  -7.0, -10.0,  -5.0, -10.0,   7.0,
    ]);

    let svd = svd_golub_kahan(&a);

    let recon = reconstruct(&svd);
    let err = (recon - a).norm_fro();
    assert!(err < 1e-4, "10x10 GK reconstruction error: {}", err);
}
