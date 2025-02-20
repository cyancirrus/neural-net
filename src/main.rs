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



fn lu_factorization(x:&blas::NdArray) -> (NdArray, NdArray) {
    let rows = x.dims[0];
    let cols = x.dims[1];
    assert_eq!(rows, cols, "currently LU is available only for square");
    let mut lower = vec![0_f32;x.data.len()]; 
    let mut upper = x.data.clone();

    for j in 0..rows {
        for i in 0..rows {
            for k in 0..rows {
                if j > i && k == 0 {
                    upper[j * cols + i ] = 0_f32;
                } else if i == j && k == 0 {
                    lower[i * cols + j] = 1_f32;
                } else if i > j {
                    if k == 0 {
                        lower[ i * cols + j] = -upper[i * cols + j] / upper[j * cols + j];
                        upper[ i * cols + j] = 0_f32;
                    } else {
                        upper[i * cols + k] += lower[i * cols + j] * upper[j * cols + k];
                    }
                }
            }
        }
    }
    (
        blas::NdArray::new(x.dims.clone(), lower),
        blas::NdArray::new(x.dims.clone(), upper),
    )
}
                    
                    // let lower_index = i * cols + j;
                    // let upper_index = j * cols + k;
                    // let target_index = i * cols + k;
// if target_index / cols + 1 == 3 {
//                     // {
//                         println!("---------------------------");
//                         println!("target: ({},{}), lower: ({},{}) upper: ({},{})", 
//                             target_index / cols + 1, target_index % cols + 1,
//                             lower_index / cols + 1, lower_index % cols + 1,
//                             upper_index / cols + 1, upper_index % cols + 1);
//                         println!("i:{}, j:{}, k:{}, update:{}, curr:{}", i, j, k, value, upper[target_index]);
//                     }


fn main () {
    let mut data = vec![0_f32;16];
    let dims = vec![4;2];
    for i in 0..8 {
        data[i] = i as f32 + 1_f32;
    }
    data[8] = 1_f32;
    data[9] = 1_f32;
    data[10] = 3_f32;
    data[11] = 3_f32;
    data[12] = 2_f32;
    data[13] = 1_f32;
    data[14] = 1_f32;
    data[15] = 1_f32;
    let ndarray = blas::NdArray::new(dims, data);
    // println!("test {:?}", ndarray);

    let fact = lu_factorization(&ndarray);
    println!("Lower: {:?}", fact.0);
    println!("Upper: {:?}", fact.1);
}
