#![allow(warnings)]
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
mod math;
mod simd;
use rand::Rng;
mod blas;
use rayon::prelude::ParallelIterator;
use rayon::prelude::*;
use blas::{NdArray, Matrix};



fn generate_matrix(rows: usize, cols: usize) -> NdArray {
    let mut rng = rand::thread_rng();
    let data: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(-10.0..10.0)).collect();
    NdArray::new(vec![rows, cols], data)
}

fn main() {
    let i = 2;  // Number of rows in x
    let j = 3;  // Number of columns in x, also rows in y
    let k = 8;  // Number of columns in y

    let x = generate_matrix(i, j);
    let y = generate_matrix(j, k);

    let blocksize = 8; // Test with different block sizes
    let result = simd::simd_tensor_mult(blocksize, x, y);

    println!("{:?}", result);

    // let x = NdArray::new(vec![2, 3], vec![
    //     1.0, 2.0, 3.0,
    //     4.0, 5.0, 6.0,
    // ]);

    // let y = NdArray::new(vec![3, 2], vec![
    //     7.0, 8.0,
    //     9.0, 10.0,
    //     11.0, 12.0,
    // ]);

    // let result = blas::parallel_tensor_mult(blocksize, x, y);
    // println!("{:?}", result);

}



// fn main() {
//     let length = 9;
//     let mut x = vec![0_f32;length];
//     let mut y = vec![0_f32;length];
//     let mut rng = rand::thread_rng();
//     for i in 0..length {
//         x[i] = rng.gen_range(-10_f32..10_f32);
//         y[i] = rng.gen_range(-10_f32..10_f32);
//     }

//     println!("Math vector product: {:?}", math::dot_product(&x,&y));
//     println!("Simd vector product: {:?}", simd::simd_dot_product(&x,&y));

// }
