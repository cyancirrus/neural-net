use neural_net::blas;
use neural_net::math;
use neural_net::simd;
use rand::Rng;



fn generate_matrix(rows: usize, cols: usize) -> NdArray {
    let mut rng = rand::thread_rng();
    let data: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(-10.0..10.0)).collect();
    NdArray::new(vec![rows, cols], data)
}

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark(c: &mut Criterion) {
    let small_size = 64;
    let medium_size = 512;
    let large_size = 1024;
    let blocksize = 16; // Example blocksize, modify as needed

    let small_x = generate_matrix(small_size, small_size);
    let small_y = generate_matrix(small_size, small_size);
    let medium_x = generate_matrix(medium_size, medium_size);
    let medium_y = generate_matrix(medium_size, medium_size);
    let large_x = generate_matrix(large_size, large_size);
    let large_y = generate_matrix(large_size, large_size);

    // Small matrix tests
    c.bench_function("small_parallel_tensor_mult", |b| {
        b.iter(|| blas::parallel_tensor_mult(blocksize, black_box(&small_x), black_box(&small_y)))
    });
    c.bench_function("small_tensor_mult", |b| {
        b.iter(|| blas::tensor_mult(blocksize, black_box(&small_x), black_box(&small_y)))
    });
    c.bench_function("small_simd_tensor_mult", |b| {
        b.iter(|| simd::simd_tensor_mult(8, black_box(&small_x), black_box(&small_y)))
    });

    // Medium matrix tests
    c.bench_function("medium_parallel_tensor_mult", |b| {
        b.iter(|| blas::parallel_tensor_mult(blocksize, black_box(&medium_x), black_box(&medium_y)))
    });
    c.bench_function("medium_tensor_mult", |b| {
        b.iter(|| blas::tensor_mult(blocksize, black_box(&medium_x), black_box(&medium_y)))
    });
    c.bench_function("medium_simd_tensor_mult", |b| {
        b.iter(|| simd::simd_tensor_mult(8, black_box(&medium_x), black_box(&medium_y)))
    });

    // Large matrix tests
    c.bench_function("large_parallel_tensor_mult", |b| {
        b.iter(|| blas::parallel_tensor_mult(blocksize, black_box(&large_x), black_box(&large_y)))
    });
    c.bench_function("large_tensor_mult", |b| {
        b.iter(|| blas::tensor_mult(blocksize, black_box(&large_x), black_box(&large_y)))
    });
    c.bench_function("large_simd_tensor_mult", |b| {
        b.iter(|| simd::simd_tensor_mult(8, black_box(&large_x), black_box(&large_y)))
    });
}

criterion_group!(benches, benchmark);
criterion_main!(benches);

