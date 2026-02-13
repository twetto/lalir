use std::fmt;
use std::ops::{Add, Sub, Mul, Index, IndexMut};

/// A statically-sized matrix stored on the stack in row-major order.
/// M = rows, N = cols.
#[derive(Clone, Copy)]
pub struct Matrix<const M: usize, const N: usize> {
    pub data: [[f64; N]; M],
}

// --- Creation ---

impl<const M: usize, const N: usize> Matrix<M, N> {
    /// Zero matrix
    pub fn zeros() -> Self {
        Self { data: [[0.0; N]; M] }
    }

    /// Create from a 2D array
    pub fn from_array(data: [[f64; N]; M]) -> Self {
        Self { data }
    }

    /// Create from a flat slice (row-major), matching C's fill_matrix
    pub fn from_slice(values: &[f64]) -> Self {
        assert!(values.len() >= M * N, "Not enough values");
        let mut mat = Self::zeros();
        for i in 0..M {
            for j in 0..N {
                mat.data[i][j] = values[i * N + j];
            }
        }
        mat
    }

    pub fn rows(&self) -> usize { M }
    pub fn cols(&self) -> usize { N }

    /// Transpose: M×N -> N×M
    pub fn transpose(&self) -> Matrix<N, M> {
        let mut t = Matrix::<N, M>::zeros();
        for i in 0..M {
            for j in 0..N {
                t.data[j][i] = self.data[i][j];
            }
        }
        t
    }

    /// Scalar multiplication: B = alpha * A
    pub fn scale(&self, alpha: f64) -> Self {
        let mut b = *self;
        for i in 0..M {
            for j in 0..N {
                b.data[i][j] *= alpha;
            }
        }
        b
    }

    /// Scalar addition: B = A + beta (elementwise)
    pub fn add_scalar(&self, beta: f64) -> Self {
        let mut b = *self;
        for i in 0..M {
            for j in 0..N {
                b.data[i][j] += beta;
            }
        }
        b
    }

    /// Matrix-vector product: w = A * v
    /// A is M×N, v is length N, result is length M
    pub fn matvec(&self, v: &Vector<N>) -> Vector<M> {
        let mut w = Vector::<M>::zeros();
        for i in 0..M {
            let mut sum = 0.0;
            for j in 0..N {
                sum += self.data[i][j] * v.data[j];
            }
            w.data[i] = sum;
        }
        w
    }

    /// Frobenius norm
    pub fn norm_fro(&self) -> f64 {
        let mut sum = 0.0;
        for i in 0..M {
            for j in 0..N {
                sum += self.data[i][j] * self.data[i][j];
            }
        }
        sum.sqrt()
    }
}

// --- Square matrix operations ---

impl<const N: usize> Matrix<N, N> {
    /// Identity matrix
    pub fn identity() -> Self {
        let mut mat = Self::zeros();
        for i in 0..N {
            mat.data[i][i] = 1.0;
        }
        mat
    }

    /// Maximum absolute diagonal value
    pub fn max_abs_diag(&self) -> f64 {
        let mut max_val = 0.0f64;
        for i in 0..N {
            max_val = max_val.max(self.data[i][i].abs());
        }
        max_val
    }
}

// --- Indexing: mat[(i, j)] ---

impl<const M: usize, const N: usize> Index<(usize, usize)> for Matrix<M, N> {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.data[i][j]
    }
}

impl<const M: usize, const N: usize> IndexMut<(usize, usize)> for Matrix<M, N> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        &mut self.data[i][j]
    }
}

// --- Matrix arithmetic operators ---

/// Elementwise addition: C = A + B
impl<const M: usize, const N: usize> Add for Matrix<M, N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut c = Self::zeros();
        for i in 0..M {
            for j in 0..N {
                c.data[i][j] = self.data[i][j] + rhs.data[i][j];
            }
        }
        c
    }
}

/// Elementwise subtraction: C = A - B
impl<const M: usize, const N: usize> Sub for Matrix<M, N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let mut c = Self::zeros();
        for i in 0..M {
            for j in 0..N {
                c.data[i][j] = self.data[i][j] - rhs.data[i][j];
            }
        }
        c
    }
}

/// Matrix multiplication: C = A * B (i-k-j loop order for cache efficiency)
/// A is M×K, B is K×N, result is M×N
impl<const M: usize, const K: usize, const N: usize> Mul<Matrix<K, N>> for Matrix<M, K> {
    type Output = Matrix<M, N>;
    fn mul(self, rhs: Matrix<K, N>) -> Matrix<M, N> {
        let mut c = Matrix::<M, N>::zeros();
        for i in 0..M {
            for k in 0..K {
                let a_ik = self.data[i][k];
                for j in 0..N {
                    c.data[i][j] += a_ik * rhs.data[k][j];
                }
            }
        }
        c
    }
}

/// Scalar * Matrix: 2.0 * A
impl<const M: usize, const N: usize> Mul<Matrix<M, N>> for f64 {
    type Output = Matrix<M, N>;
    fn mul(self, rhs: Matrix<M, N>) -> Matrix<M, N> {
        rhs.scale(self)
    }
}

/// Matrix * Scalar: A * 2.0
impl<const M: usize, const N: usize> Mul<f64> for Matrix<M, N> {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        self.scale(rhs)
    }
}

// --- Display ---

impl<const M: usize, const N: usize> fmt::Display for Matrix<M, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..M {
            for j in 0..N {
                write!(f, "{:8.4} ", self.data[i][j])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

// ============================================================
// Vector<N>: statically-sized vector on the stack
// ============================================================

#[derive(Clone, Copy)]
pub struct Vector<const N: usize> {
    pub data: [f64; N],
}

impl<const N: usize> Vector<N> {
    pub fn zeros() -> Self {
        Self { data: [0.0; N] }
    }

    pub fn from_array(data: [f64; N]) -> Self {
        Self { data }
    }

    /// Dot product
    pub fn dot(&self, other: &Self) -> f64 {
        let mut sum = 0.0;
        for i in 0..N {
            sum += self.data[i] * other.data[i];
        }
        sum
    }

    /// L2 norm
    pub fn norm(&self) -> f64 {
        self.dot(self).sqrt()
    }

    /// Return a normalized copy, or None if zero vector
    pub fn normalized(&self) -> Option<Self> {
        let n = self.norm();
        if n == 0.0 {
            return None;
        }
        let mut out = *self;
        for i in 0..N {
            out.data[i] /= n;
        }
        Some(out)
    }

    /// Outer product: u ⊗ v → N×P matrix
    pub fn outer<const P: usize>(&self, other: &Vector<P>) -> Matrix<N, P> {
        let mut mat = Matrix::<N, P>::zeros();
        for i in 0..N {
            for j in 0..P {
                mat.data[i][j] = self.data[i] * other.data[j];
            }
        }
        mat
    }
}

// --- Vector arithmetic operators ---

impl<const N: usize> Add for Vector<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut out = self;
        for i in 0..N {
            out.data[i] += rhs.data[i];
        }
        out
    }
}

impl<const N: usize> Sub for Vector<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let mut out = self;
        for i in 0..N {
            out.data[i] -= rhs.data[i];
        }
        out
    }
}

/// Scalar * Vector: a * v
impl<const N: usize> Mul<Vector<N>> for f64 {
    type Output = Vector<N>;
    fn mul(self, rhs: Vector<N>) -> Vector<N> {
        let mut out = rhs;
        for i in 0..N {
            out.data[i] *= self;
        }
        out
    }
}

/// Vector * Scalar: v * a
impl<const N: usize> Mul<f64> for Vector<N> {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        rhs * self
    }
}

// --- Vector indexing ---

impl<const N: usize> Index<usize> for Vector<N> {
    type Output = f64;
    fn index(&self, i: usize) -> &f64 {
        &self.data[i]
    }
}

impl<const N: usize> IndexMut<usize> for Vector<N> {
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        &mut self.data[i]
    }
}

impl<const N: usize> fmt::Display for Vector<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for i in 0..N {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{:.4}", self.data[i])?;
        }
        write!(f, "]")
    }
}

/// Cross product (3D only)
pub fn cross(a: &Vector<3>, b: &Vector<3>) -> Vector<3> {
    Vector::from_array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])
}
