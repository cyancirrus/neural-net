#![allow(warnings)]
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use crate::calc_utils::blas;

pub fn supports_avx2() -> bool {
    is_x86_feature_detected!("avx2")
}

pub fn supports_sse4() -> bool {
    is_x86_feature_detected!("sse4.1")
}

pub fn simd_vector_add(x: &[f32], y: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), y.len());
    let length = x.len();
    let mut result = vec![0_f32; length]; 
    let chunks = length / 8;
    let remainder = length % 8;

    unsafe {
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let res_ptr = result.as_mut_ptr();

        for i in 0..chunks {
            let x_chunk = _mm256_loadu_ps(x_ptr.add(i*8));
            let y_chunk = _mm256_loadu_ps(y_ptr.add(i*8));
            let sum_chunk = _mm256_add_ps(x_chunk, y_chunk);
            _mm256_storeu_ps(res_ptr.add(i * 8), sum_chunk);
        }
        for i in length - remainder..length {
            result[i] = x[i] + y[i];
        }
    }
    result
}

pub fn simd_vector_diff(x: &[f32], y: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), y.len());
    let length = x.len();
    let mut result = vec![0_f32;length];
    let chunks = length / 8;
    let remainder = length % 8;
    
    unsafe {
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let res_ptr = result.as_mut_ptr();

        for i in 0..chunks {
            let x_chunk = _mm256_loadu_ps(x_ptr.add(i*8));
            let y_chunk = _mm256_loadu_ps(y_ptr.add(i*8));
            let sum_chunk = _mm256_sub_ps(x_chunk, y_chunk);
            _mm256_storeu_ps(res_ptr.add(i*8), sum_chunk);
        }
        for i in length - remainder..length {
            result[i] = x[i] - y[i];
        }
    }
    result
}

pub fn simd_vector_product(x:&[f32], y:&[f32]) -> Vec<f32> {
    assert_eq!(x.len(), y.len());
    let length = x.len();
    let mut result = vec![0_f32;length];
    let blocks = length / 8;
    let remainder = length % 8;
    
    unsafe {
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let res_ptr = result.as_mut_ptr();

        for i in 0..blocks {
            let x_chunk = _mm256_loadu_ps(x_ptr.add(i*8));
            let y_chunk = _mm256_loadu_ps(y_ptr.add(i*8));
            let sum_chunk = _mm256_mul_ps(x_chunk,y_chunk);
            _mm256_storeu_ps(res_ptr.add(i*8), sum_chunk);
        }
        for i in length - remainder..length {
            result[i] = x[i] - y[i];
        }
    }
    result
}

pub fn simd_dot_product(x:&[f32], y:&[f32]) -> f32 {
    assert_eq!(x.len(), y.len());
    let length = x.len();
    let mut result:f32 = 0_f32;
    let blocks = length / 8;
    let remainder = length % 8;
    
    unsafe {
        let mut sum_vec = _mm256_setzero_ps();
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        for i in 0..blocks {
            let x_chunk = _mm256_loadu_ps(x_ptr.add(i*8));
            let y_chunk = _mm256_loadu_ps(y_ptr.add(i*8));
            let prod_chunk = _mm256_mul_ps(x_chunk, y_chunk);
            sum_vec = _mm256_add_ps(sum_vec, prod_chunk);
        }

        let sum_arr = std::mem::transmute::<_,[f32;8]>( sum_vec );
        result = sum_arr.iter().sum::<f32>();
        
        for i in length - remainder..length {
            result += x[i] * y[i]
        }
    }
    result
}

pub fn simd_tensor_mult(blocksize: usize, x: &blas::NdArray, y: &blas::NdArray) -> blas::NdArray {
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
                        let result = simd_dot_product(
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
    let mut dims = x.dims.clone();
    dims[1] = y.dims[1];
    blas::NdArray::new(dims, new)
}
