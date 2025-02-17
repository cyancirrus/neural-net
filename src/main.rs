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

fn simd_tensor_mult(blocksize: usize, x: NdArray, y: NdArray) -> NdArray {
    assert!(blocksize == 8);  // Ensure blocksize is 8 for SIMD
    assert_eq!(x.dims[1], y.dims[0], "dimension mismatch");

    let x_rows = x.dims[0];
    let x_cols = x.dims[1];
    let y_cols = y.dims[1];
    let mut new: Vec<f32> = vec![0_f32; x_rows * y_cols];

    // Loop over blocks
    for i in (0..x_rows).step_by(blocksize) {
        for j in (0..y_cols).step_by(blocksize) {
            for k in 0..(x_cols + blocksize - 1) / blocksize {
                // Loop over rows and columns in blocks
                for ii in 0..blocksize.min(x_rows - i) {
                    for jj in 0..blocksize.min(y_cols - j) {
                        // Calculate available length for this block to avoid accessing out-of-bounds memory
                        let available = blocksize.min(x_cols - k * blocksize);
                        
                        let x_index = (i + ii ) * x_cols + k * blocksize;
                        let y_index = (k * blocksize) * y_cols + jj + j;

                        // Generate the slice for `y`
                        let y_slice:Vec<f32> = y.data
                            .iter()
                            .skip(y_index)
                            .step_by(y_cols)
                            .take(available)
                            .map(|&val| val)  // Dereference to get `f32`
                            .collect();


                        // Perform SIMD dot product for this block
                        let result = simd::simd_dot_product(
                            &x.data[x_index..x_index + available],
                            // &y_slice.collect::<Vec<f32>>(),
                            &y_slice,
                        );

                        // Store the result in the new matrix at the appropriate index
                        let index = (i + ii) * y_cols + jj + j;
                        new[index] += result;
                    }
                }
            }
        }
    }

    // Return the new NdArray with updated dimensions
    let mut dims = x.dims.clone();
    dims[1] = y.dims[1];
    NdArray::new(dims, new)
}

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
