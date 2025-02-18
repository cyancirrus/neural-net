#![allow(warnings)]
use blas::{Matrix, NdArray};
use neural_net::calc_utils::blas;
use neural_net::calc_utils::math;
use neural_net::calc_utils::simd;
use rand::Rng;
use rayon::prelude::ParallelIterator;
use rayon::prelude::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

fn generate_matrix(rows: usize, cols: usize) -> NdArray {
    let mut rng = rand::thread_rng();
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| rng.gen_range(-10.0..10.0))
        .collect();
    NdArray::new(vec![rows, cols], data)
}

// fn main () {
//     println!("hello world!");
// }

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark(c: &mut Criterion) {
    // let small_size = 64;
    // let medium_size = 512;
    // let large_size = 1024;
    let small_size = 64;
    let medium_size = 128;
    let large_size = 256;
    let blocksize = 16; // Example blocksize, modify as needed
                        //

    let small_x = generate_matrix(small_size, small_size);
    let small_y = generate_matrix(small_size, small_size);
    let medium_x = generate_matrix(medium_size, medium_size);
    let medium_y = generate_matrix(medium_size, medium_size);
    let large_x = generate_matrix(large_size, large_size);
    let large_y = generate_matrix(large_size, large_size);

    blas::parallel_tensor_mult(blocksize, &small_x, &small_y);
    blas::parallel_tensor_mult(blocksize, &small_x, &small_y);
    blas::parallel_tensor_mult(blocksize, &small_x, &small_y);
    blas::parallel_tensor_mult(blocksize, &small_x, &small_y);
    blas::parallel_tensor_mult(blocksize, &medium_x, &medium_y);
    blas::parallel_tensor_mult(blocksize, &medium_x, &medium_y);
    blas::parallel_tensor_mult(blocksize, &medium_x, &medium_y);
    blas::parallel_tensor_mult(blocksize, &medium_x, &medium_y);
    blas::parallel_tensor_mult(blocksize, &large_x, &large_y);
    blas::parallel_tensor_mult(blocksize, &large_x, &large_y);
    blas::parallel_tensor_mult(blocksize, &large_x, &large_y);
    blas::parallel_tensor_mult(blocksize, &large_x, &large_y);
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
