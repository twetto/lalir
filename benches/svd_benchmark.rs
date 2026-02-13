use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lalir::matrix::Matrix;
use lalir::svd::svd_augmented;
use lalir::svd_gk::svd_golub_kahan;

// Helper to generate a "random-ish" matrix deterministically
// (We don't want to benchmark the random number generator!)
fn generate_matrix<const M: usize, const N: usize>() -> Matrix<M, N> {
    let mut mat = Matrix::zeros();
    for i in 0..M {
        for j in 0..N {
            // Use sin/cos to make non-trivial numbers
            let val = (i as f64 * 3.5 + j as f64 * 2.1).sin() * 10.0;
            mat[(i, j)] = val;
        }
    }
    mat
}

fn bench_svd_16x16(c: &mut Criterion) {
    let mut group = c.benchmark_group("SVD 16x16");

    // Create the input data ONCE, outside the loop
    let a = generate_matrix::<16, 16>();

    // 1. Benchmark Golub-Kahan
    group.bench_function("Golub-Kahan", |b| {
        b.iter(|| {
            // black_box prevents the compiler from optimizing the result away
            svd_golub_kahan(black_box(&a))
        })
    });

    // 2. Benchmark Jacobi
    // Note: MN = 16 + 16 = 32
    group.bench_function("Jacobi (Augmented)", |b| {
        b.iter(|| {
            svd_augmented::<16, 16, 32>(black_box(&a))
        })
    });

    group.finish();
}

fn bench_svd_4x4(c: &mut Criterion) {
    let mut group = c.benchmark_group("SVD 4x4");
    let a = generate_matrix::<4, 4>();

    group.bench_function("Golub-Kahan", |b| {
        b.iter(|| svd_golub_kahan(black_box(&a)))
    });

    // MN = 4 + 4 = 8
    group.bench_function("Jacobi (Augmented)", |b| {
        b.iter(|| svd_augmented::<4, 4, 8>(black_box(&a)))
    });

    group.finish();
}

criterion_group!(benches, bench_svd_4x4, bench_svd_16x16);
criterion_main!(benches);
