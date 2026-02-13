# LALIR (Linear Algebra Library In Rust)

**LALIR** is a (mostly) statically-sized linear algebra library written in Rust for educational use. 

Instead of allocating on the heap (like `Vec<f64>`), LALIR uses **Const Generics** to store matrices and vectors directly on the stack. This results in very little heap allocation for core operations, rigorous compile-time size checking, and good cache locality.

## Features

### Core Structures
- **Stack-Allocated:** `Matrix<M, N>` and `Vector<N>` types backed by pure arrays (`[[f64; N]; M]`).
- **Safe Initialization:** `zeros()`, `identity()`, `from_slice()`, `from_array()`.
- **Compile-Time Checks:** Dimension mismatches (e.g., multiplying 3x3 by 4x4) are caught at compile time.

### Decompositions & Solvers
LALIR implements a comprehensive suite of numerical algorithms:

* **LU Decomposition**: Partial pivoting ($PA = LU$) for solving square systems.
* **QR Decomposition**:
    * Householder (Compact storage)
    * Givens Rotations
    * Gram-Schmidt
    * **Column Pivoting** ($AP = QR$) for rank-deficient least squares.
* **Cholesky**: Standard ($LL^T$) and LDLT ($LDL^T$) for symmetric positive definite matrices.
* **Eigen decomposition**:
    * QR Iteration (Hessenberg reduction implicitly handled for symmetric matrices).
    * Jacobi Iteration.
* **SVD (Singular Value Decomposition)**:
    * **Golub-Kahan Bidiagonalization** with implicit QR shifts (Standard industrial approach).
    * Jacobi SVD on augmented matrices.

## Usage

Add `lalir` to your project (or clone this repo).

```rust
use lalir::matrix::{Matrix, Vector};
use lalir::lu::lu_decompose;

fn main() {
    // 1. Create matrices on the stack
    let a = Matrix::<3, 3>::from_slice(&[
        2.0, -1.0, 0.0,
        -1.0, 2.0, -1.0,
        0.0, -1.0, 2.0
    ]);
    
    let b = Vector::<3>::from_array([1.0, 0.0, 1.0]);

    // 2. Solve Ax = b using LU Decomposition
    match lu_decompose(&a) {
        Ok(lu) => {
            let x = lu.solve(&b);
            println!("Solution x: {}", x);
        },
        Err(e) => eprintln!("Decomposition failed: {}", e),
    }

    // 3. Chain operations safely
    let At_A = a.transpose() * a;
    println!("Norm: {}", At_A.norm_fro());
}
```

## Testing

Run the test suite to verify the numerical stability of decompositions:

```bash
cargo test --tests
```

## License

MIT

