use lalir::matrix::*;

fn test_basic_operations() {
    println!("\n=== Basic Operations Test ===");

    let a = Matrix::<2, 3>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    println!("Matrix A:\n{}", a);

    let at = a.transpose();
    println!("A Transposed:\n{}", at);

    println!("A * A^T:\n{}", a * at);
    println!("A^T * A:\n{}", at * a);
    println!("2 * A:\n{}", a.scale(2.0));
    println!("A + 10:\n{}", a.add_scalar(10.0));

    let v1 = Vector::<3>::from_array([1.0, 2.0, 3.0]);
    let v2 = Vector::<3>::from_array([4.0, 5.0, 6.0]);
    println!("v1: {}", v1);
    println!("v2: {}", v2);
    println!("\nDot product ?v1, v2?: {:.4}", v1.dot(&v2));
    println!("||v1||_2: {:.4}", v1.norm());
    println!("Normalized v1: {}", v1.normalized().unwrap());

    println!("\nA * v1:\n{}", a.matvec(&v1));
    println!("\nOuter product v1 * v2^T:\n{}", v1.outer(&v2));
    println!("3x3 Identity Matrix:\n{}", Matrix::<3, 3>::identity());
}

fn main() {
    test_basic_operations();
}
