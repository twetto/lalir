use lalir::matrix::*;

const TOL: f64 = 1e-10;

fn assert_mat_eq<const M: usize, const N: usize>(
    a: &Matrix<M, N>, b: &Matrix<M, N>, label: &str
) {
    for i in 0..M {
        for j in 0..N {
            assert!(
                (a[(i,j)] - b[(i,j)]).abs() < TOL,
                "{}: mismatch at ({},{}): {} vs {}", label, i, j, a[(i,j)], b[(i,j)]
            );
        }
    }
}

fn assert_vec_eq<const N: usize>(a: &Vector<N>, b: &Vector<N>, label: &str) {
    for i in 0..N {
        assert!(
            (a[i] - b[i]).abs() < TOL,
            "{}: mismatch at {}: {} vs {}", label, i, a[i], b[i]
        );
    }
}

#[test]
fn test_transpose() {
    let a = Matrix::<2, 3>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let at = a.transpose();
    let expected = Matrix::<3, 2>::from_slice(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert_mat_eq(&at, &expected, "A^T");
}

#[test]
fn test_matmul_aat() {
    // A * A^T  (2x3 * 3x2 = 2x2)
    let a = Matrix::<2, 3>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let at = a.transpose();
    let c = a * at;
    let expected = Matrix::<2, 2>::from_slice(&[14.0, 32.0, 32.0, 77.0]);
    assert_mat_eq(&c, &expected, "A * A^T");
}

#[test]
fn test_matmul_ata() {
    // A^T * A  (3x2 * 2x3 = 3x3)
    let a = Matrix::<2, 3>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let at = a.transpose();
    let d = at * a;
    let expected = Matrix::<3, 3>::from_slice(&[
        17.0, 22.0, 27.0,
        22.0, 29.0, 36.0,
        27.0, 36.0, 45.0,
    ]);
    assert_mat_eq(&d, &expected, "A^T * A");
}

#[test]
fn test_scalar_mul() {
    let a = Matrix::<2, 3>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let scaled = a.scale(2.0);
    let expected = Matrix::<2, 3>::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    assert_mat_eq(&scaled, &expected, "2*A");
}

#[test]
fn test_scalar_add() {
    let a = Matrix::<2, 3>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let added = a.add_scalar(10.0);
    let expected = Matrix::<2, 3>::from_slice(&[11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
    assert_mat_eq(&added, &expected, "A + 10");
}

#[test]
fn test_elementwise_add_sub() {
    let a = Matrix::<2, 2>::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let b = Matrix::<2, 2>::from_slice(&[5.0, 6.0, 7.0, 8.0]);
    let sum = a + b;
    let diff = a - b;
    let expected_sum = Matrix::<2, 2>::from_slice(&[6.0, 8.0, 10.0, 12.0]);
    let expected_diff = Matrix::<2, 2>::from_slice(&[-4.0, -4.0, -4.0, -4.0]);
    assert_mat_eq(&sum, &expected_sum, "A + B");
    assert_mat_eq(&diff, &expected_diff, "A - B");
}

#[test]
fn test_dot_product() {
    let v1 = Vector::<3>::from_array([1.0, 2.0, 3.0]);
    let v2 = Vector::<3>::from_array([4.0, 5.0, 6.0]);
    let dot = v1.dot(&v2);
    assert!((dot - 32.0).abs() < TOL, "dot product: {} vs 32.0", dot);
}

#[test]
fn test_l2_norm() {
    let v1 = Vector::<3>::from_array([1.0, 2.0, 3.0]);
    let norm = v1.norm();
    let expected = (14.0f64).sqrt();
    assert!((norm - expected).abs() < TOL, "||v1||: {} vs {}", norm, expected);
}

#[test]
fn test_normalize() {
    let v1 = Vector::<3>::from_array([1.0, 2.0, 3.0]);
    let n = v1.normalized().expect("Should not be zero");
    let norm = (14.0f64).sqrt();
    let expected = Vector::<3>::from_array([1.0 / norm, 2.0 / norm, 3.0 / norm]);
    assert_vec_eq(&n, &expected, "normalized v1");
    // Normalized vector should have unit length
    assert!((n.norm() - 1.0).abs() < TOL, "||normalized|| should be 1.0");
}

#[test]
fn test_normalize_zero() {
    let v = Vector::<3>::zeros();
    assert!(v.normalized().is_none(), "Zero vector should return None");
}

#[test]
fn test_matvec() {
    let a = Matrix::<2, 3>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let v = Vector::<3>::from_array([1.0, 2.0, 3.0]);
    let w = a.matvec(&v);
    let expected = Vector::<2>::from_array([14.0, 32.0]);
    assert_vec_eq(&w, &expected, "A * v1");
}

#[test]
fn test_outer_product() {
    let v1 = Vector::<3>::from_array([1.0, 2.0, 3.0]);
    let v2 = Vector::<3>::from_array([4.0, 5.0, 6.0]);
    let outer = v1.outer(&v2);
    let expected = Matrix::<3, 3>::from_slice(&[
         4.0,  5.0,  6.0,
         8.0, 10.0, 12.0,
        12.0, 15.0, 18.0,
    ]);
    assert_mat_eq(&outer, &expected, "v1 ? v2");
}

#[test]
fn test_identity() {
    let i = Matrix::<3, 3>::identity();
    let expected = Matrix::<3, 3>::from_slice(&[
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ]);
    assert_mat_eq(&i, &expected, "I_3");
}

#[test]
fn test_identity_multiply() {
    // A * I = A
    let a = Matrix::<2, 3>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let i3 = Matrix::<3, 3>::identity();
    let ai = a * i3;
    assert_mat_eq(&ai, &a, "A * I = A");
}

#[test]
fn test_cross_product() {
    let a = Vector::<3>::from_array([1.0, 0.0, 0.0]);
    let b = Vector::<3>::from_array([0.0, 1.0, 0.0]);
    let c = cross(&a, &b);
    let expected = Vector::<3>::from_array([0.0, 0.0, 1.0]);
    assert_vec_eq(&c, &expected, "x cross y = z");
}

#[test]
fn test_indexing() {
    let mut m = Matrix::<2, 2>::zeros();
    m[(0, 0)] = 1.0;
    m[(0, 1)] = 2.0;
    m[(1, 0)] = 3.0;
    m[(1, 1)] = 4.0;
    assert!((m[(0, 0)] - 1.0).abs() < TOL);
    assert!((m[(1, 1)] - 4.0).abs() < TOL);
}

#[test]
fn test_frobenius_norm() {
    let a = Matrix::<2, 2>::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let nf = a.norm_fro();
    let expected = (30.0f64).sqrt(); // 1+4+9+16
    assert!((nf - expected).abs() < TOL, "Frobenius norm: {} vs {}", nf, expected);
}

#[test]
fn test_scalar_mul_operators() {
    let a = Matrix::<2, 2>::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let expected = Matrix::<2, 2>::from_slice(&[2.0, 4.0, 6.0, 8.0]);
    assert_mat_eq(&(a * 2.0), &expected, "A * 2.0");
    assert_mat_eq(&(2.0 * a), &expected, "2.0 * A");
}

#[test]
fn test_vector_scalar_mul() {
    let v = Vector::<3>::from_array([1.0, 2.0, 3.0]);
    let expected = Vector::<3>::from_array([3.0, 6.0, 9.0]);
    assert_vec_eq(&(v * 3.0), &expected, "v * 3.0");
    assert_vec_eq(&(3.0 * v), &expected, "3.0 * v");
}

/// Runs all tests as a visual print (like the C version's main output)
fn main() {
    println!("=== Basic Operations Test ===\n");

    let a = Matrix::<2, 3>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    println!("Matrix A:\n{}", a);

    let at = a.transpose();
    println!("A Transposed:\n{}", at);

    let c = a * at;
    println!("A * A^T:\n{}", c);

    let d = at * a;
    println!("A^T * A:\n{}", d);

    let scaled = a.scale(2.0);
    println!("2 * A:\n{}", scaled);

    let added = a.add_scalar(10.0);
    println!("A + 10:\n{}", added);

    let v1 = Vector::<3>::from_array([1.0, 2.0, 3.0]);
    let v2 = Vector::<3>::from_array([4.0, 5.0, 6.0]);
    println!("v1: {}", v1);
    println!("v2: {}", v2);
    println!("Dot product <v1, v2>: {:.4}", v1.dot(&v2));
    println!("||v1||_2: {:.4}", v1.norm());
    println!("Normalized v1: {}", v1.normalized().unwrap());

    let w = a.matvec(&v1);
    println!("\nA * v1: {}", w);

    let outer = v1.outer(&v2);
    println!("\nOuter product v1 * v2^T:\n{}", outer);

    let i = Matrix::<3, 3>::identity();
    println!("3x3 Identity:\n{}", i);
}
