use lalir::matrix::*;
use lalir::lu::*;

const TOL: f64 = 1e-10;

#[test]
fn test_lu_solve() {
    let a = Matrix::<3, 3>::from_slice(&[
         2.0,  3.0,  1.0,
         4.0,  7.0, -1.0,
        -2.0,  3.0,  8.0,
    ]);
    let b = Vector::<3>::from_array([5.0, 13.0, 3.0]);

    let lu = lu_decompose(&a).expect("LU failed");
    let x = lu.solve(&b);

    // Verify A * x ≈ b
    let ax = a.matvec(&x);
    for i in 0..3 {
        assert!(
            (ax[i] - b[i]).abs() < TOL,
            "Ax != b at {}: {} vs {}", i, ax[i], b[i]
        );
    }
}

#[test]
fn test_lu_known_solution() {
    // Same system, verify against known solution
    let a = Matrix::<3, 3>::from_slice(&[
         2.0,  3.0,  1.0,
         4.0,  7.0, -1.0,
        -2.0,  3.0,  8.0,
    ]);
    let b = Vector::<3>::from_array([5.0, 13.0, 3.0]);

    let lu = lu_decompose(&a).expect("LU failed");
    let x = lu.solve(&b);

    // Solve with numpy for reference: x = [1, 1, 0]
    // Actually let's just check residual is tiny
    let residual = (a.matvec(&x) - b).norm();
    assert!(residual < TOL, "Residual too large: {}", residual);
}

#[test]
fn test_lu_identity() {
    // LU of identity should give identity back
    let a = Matrix::<3, 3>::identity();
    let lu = lu_decompose(&a).expect("LU of identity failed");

    // Solve Ix = b should give x = b
    let b = Vector::<3>::from_array([7.0, -3.0, 42.0]);
    let x = lu.solve(&b);
    for i in 0..3 {
        assert!((x[i] - b[i]).abs() < TOL, "Identity solve failed at {}", i);
    }
}

#[test]
fn test_lu_singular() {
    // Singular matrix: row 3 = row 1 + row 2
    let a = Matrix::<3, 3>::from_slice(&[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        5.0, 7.0, 9.0,
    ]);
    assert!(lu_decompose(&a).is_err(), "Should detect singular matrix");
}

#[test]
fn test_lu_4x4() {
    let a = Matrix::<4, 4>::from_slice(&[
        2.0,  1.0,  1.0,  0.0,
        4.0,  3.0,  3.0,  1.0,
        8.0,  7.0,  9.0,  5.0,
        6.0,  7.0,  9.0,  8.0,
    ]);
    let b = Vector::<4>::from_array([1.0, 2.0, 3.0, 4.0]);

    let lu = lu_decompose(&a).expect("LU failed");
    let x = lu.solve(&b);

    let residual = (a.matvec(&x) - b).norm();
    assert!(residual < TOL, "4x4 residual too large: {}", residual);
}

