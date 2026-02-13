use lalir::matrix::*;
use lalir::svd::*;

const SVD_TOL: f64 = 1e-6;

// ============================================================
// SVD via QR iteration (matching test_svd_qr)
// ============================================================

#[test]
fn test_svd_qr_3x2() {
    let a = Matrix::<3, 2>::from_slice(&[
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    ]);

    let svd = svd_qr(&a);

    // Singular values should be positive and descending
    assert!(svd.s[0] >= svd.s[1], "Singular values not sorted");
    assert!(svd.s[0] > 0.0);

    // Reconstruct: A \approx U * diag(S) * V^T
    let recon = reconstruct(&svd);
    let err = (recon - a).norm_fro();
    assert!(err < SVD_TOL, "SVD QR reconstruction error: {}", err);
}

#[test]
fn test_svd_qr_singular_values() {
    // Known singular values for [[1,2],[3,4],[5,6]]
    // \sigma_1 \approx 9.5255, \sigma_2 \approx 0.5143
    let a = Matrix::<3, 2>::from_slice(&[
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    ]);

    let svd = svd_qr(&a);
    assert!((svd.s[0] - 9.5255).abs() < 0.001, "sigma_1 = {}", svd.s[0]);
    assert!((svd.s[1] - 0.5143).abs() < 0.001, "sigma_2 = {}", svd.s[1]);
}

#[test]
fn test_svd_qr_u_orthonormal() {
    let a = Matrix::<3, 2>::from_slice(&[
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    ]);

    let svd = svd_qr(&a);

    // U^T U should be I (2x2)
    let utu = svd.u.transpose() * svd.u;
    let err = (utu - Matrix::<2, 2>::identity()).norm_fro();
    assert!(err < SVD_TOL, "U?U not identity: {}", err);
}

#[test]
fn test_svd_qr_v_orthogonal() {
    let a = Matrix::<3, 2>::from_slice(&[
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    ]);

    let svd = svd_qr(&a);

    // V^T V should be I (2x2)
    let vtv = svd.v.transpose() * svd.v;
    let err = (vtv - Matrix::<2, 2>::identity()).norm_fro();
    assert!(err < SVD_TOL, "V?V not identity: {}", err);
}

// ============================================================
// SVD via augmented Jacobi (matching test_svd_jacobi)
// ============================================================

#[test]
fn test_svd_jacobi_3x2() {
    let a = Matrix::<3, 2>::from_slice(&[
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    ]);

    // MN = M + N = 3 + 2 = 5
    let svd = svd_augmented::<3, 2, 5>(&a);

    let recon = reconstruct(&svd);
    let err = (recon - a).norm_fro();
    assert!(err < SVD_TOL, "SVD Jacobi reconstruction error: {}", err);
}

#[test]
fn test_svd_methods_agree() {
    let a = Matrix::<3, 2>::from_slice(&[
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    ]);

    let svd1 = svd_qr(&a);
    let svd2 = svd_augmented::<3, 2, 5>(&a);

    // Singular values should agree
    for i in 0..2 {
        assert!(
            (svd1.s[i] - svd2.s[i]).abs() < 0.01,
            "Singular values disagree at {}: {} vs {}", i, svd1.s[i], svd2.s[i]
        );
    }

    // Reconstructions should both be close to A
    let recon1 = reconstruct(&svd1);
    let recon2 = reconstruct(&svd2);
    let err = (recon1 - recon2).norm_fro();
    assert!(err < SVD_TOL, "Reconstructions differ: {}", err);
}

// ============================================================
// Fat matrix (matching test_svd_qr_fat)
// ============================================================

#[test]
fn test_svd_fat_2x3() {
    let a = Matrix::<2, 3>::from_slice(&[
        1.0, 3.0, 5.0,
        2.0, 4.0, 6.0,
    ]);

    let svd = svd_fat(&a);

    // Should have 2 singular values (R = min(2,3) = 2)
    assert!(svd.s[0] >= svd.s[1], "Not sorted");

    let recon = reconstruct(&svd);
    let err = (recon - a).norm_fro();
    assert!(err < SVD_TOL, "Fat SVD reconstruction error: {}", err);
}

// ============================================================
// Square matrix (matching test_svd_qr)
// ============================================================

#[test]
fn test_svd_square_3x3() {
    let a = Matrix::<3, 3>::from_slice(&[
        12.0, -51.0,   4.0,
         6.0, 167.0, -68.0,
        -4.0,  24.0, -41.0,
    ]);

    let svd = svd_qr(&a);

    let recon = reconstruct(&svd);
    let err = (recon - a).norm_fro();
    assert!(err < SVD_TOL, "3x3 SVD reconstruction error: {}", err);
}

// ============================================================
// Larger matrix (matching test_svd_qr_large)
// ============================================================

#[test]
fn test_svd_10x10() {
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

    let svd = svd_qr(&a);

    // All singular values should be positive for this full-rank matrix
    for i in 0..10 {
        assert!(svd.s[i] > 1e-6, "sigma_{} too small: {}", i, svd.s[i]);
    }

    let recon = reconstruct(&svd);
    let err = (recon - a).norm_fro();
    assert!(err < 1e-4, "10x10 SVD reconstruction error: {}", err);
}

// ============================================================
// Rank-deficient (matching test_svd_qr_rank_deficient)
// ============================================================

#[test]
fn test_svd_rank_deficient() {
    // Row 3 = Row 1 + Row 2 => rank 2
    let a = Matrix::<3, 3>::from_slice(&[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        5.0, 7.0, 9.0,
    ]);

    let svd = svd_qr(&a);

    // Should reconstruct despite rank deficiency
    let recon = reconstruct(&svd);
    let err = (recon - a).norm_fro();
    assert!(err < SVD_TOL, "Rank-deficient reconstruction error: {}", err);

    // Numerical rank should be 2
    let rank = numerical_rank(&svd.s, 1e-6);
    assert_eq!(rank, 2, "Expected rank 2, got {}", rank);
}

#[test]
fn test_svd_rank_deficient_singular_values() {
    let a = Matrix::<3, 3>::from_slice(&[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        5.0, 7.0, 9.0,
    ]);

    let svd = svd_qr(&a);

    // Third singular value should be \approx 0
    assert!(svd.s[2] < 1e-6, "Third sigma should be ~0: {}", svd.s[2]);
    // First two should be nonzero
    assert!(svd.s[0] > 1.0, "sigma_1 too small: {}", svd.s[0]);
    assert!(svd.s[1] > 0.1, "sigma_2 too small: {}", svd.s[1]);
}

// ============================================================
// Identity matrix SVD
// ============================================================

#[test]
fn test_svd_identity() {
    let a = Matrix::<3, 3>::identity();
    let svd = svd_qr(&a);

    // All singular values should be 1
    for i in 0..3 {
        assert!(
            (svd.s[i] - 1.0).abs() < SVD_TOL,
            "Identity sigma_{} = {}", i, svd.s[i]
        );
    }

    let recon = reconstruct(&svd);
    let err = (recon - a).norm_fro();
    assert!(err < SVD_TOL, "Identity SVD reconstruction error: {}", err);
}
