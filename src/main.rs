#![allow(warnings)]
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
mod math;
use rand::Rng;
mod blas;
use rayon::prelude::ParallelIterator;
use rayon::prelude::*;
use blas::{NdArray, Matrix};

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


pub fn vector_diff(x: &[f32], y: &[f32]) -> Vec<f32> {
    x.iter().zip(y.iter()).map(|(&x, &y)| x - y).collect()
}

pub fn vector_add(x: &[f32], y: &[f32]) -> Vec<f32> {
    x.iter().zip(y.iter()).map(|(&x, &y)| x + y).collect()
}

pub fn vector_product(x: &[f32], y: &[f32]) -> Vec<f32> {
    x.iter().zip(y.iter()).map(|(&x, &y)| x * y).collect()
}

pub fn scalar_product(lambda: f32, vector: &[f32]) -> Vec<f32> {
    vector.iter().map(|&vector| lambda * vector).collect()
}


fn main() {
    let length = 9;
    let mut x = vec![0_f32;length];
    let mut y = vec![0_f32;length];
    let mut rng = rand::thread_rng();
    for i in 0..length {
        x[i] = rng.gen_range(-10_f32..10_f32);
        y[i] = rng.gen_range(-10_f32..10_f32);
    }

    println!("Math vector product: {:?}", math::vector_product(&x,&y));
    println!("Simd vector product: {:?}", simd_vector_product(&x,&y));

}
