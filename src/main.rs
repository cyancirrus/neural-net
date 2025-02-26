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

fn lu_factorization(x: &blas::NdArray) -> (NdArray, NdArray) {
    let rows = x.dims[0];
    let cols = x.dims[1];
    assert_eq!(rows, cols, "currently LU is available only for square");
    let mut lower = vec![0_f32; x.data.len()];
    let mut upper = x.data.clone();

    for j in 0..rows {
        for i in 0..rows {
            for k in 0..rows {
                if j > i && k == 0 {
                    upper[j * cols + i] = 0_f32;
                } else if i == j && k == 0 {
                    lower[i * cols + j] = 1_f32;
                } else if i > j {
                    if k == 0 {
                        lower[i * cols + j] = -upper[i * cols + j] / upper[j * cols + j];
                        upper[i * cols + j] = 0_f32;
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

fn proto_householder(mut x: &blas::NdArray) -> NdArray {
    // let magnitude = math::magnitude(&x.data);
    // let u = x.data + magnitude *
    todo!()
}

fn hoseholder_matrix(mut x: &[f32]) -> NdArray {
    let length = x.len();
    assert!(length > 0, "needs to have non-zero length");
    let dims = vec![length; 2];
    let data = vec![0_f32; length * length];
    let mut householder = NdArray::new(dims, data);
    let max_element: f32 = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut u = x.par_iter().copied()
        .map(|val| val / max_element)
        .collect::<Vec<f32>>();
    u[0] += math::magnitude(&u) * x[0].signum();
    let magnitude_squared = math::dot_product(&u, &u);
    for i in 0..length {
        // initialize the identity matrix
        householder.data[i * length + i] = 1_f32;
    }
    let projection = outer_product(u);
    householder.data.par_iter_mut()
        .zip(projection.par_iter())
        .for_each(|(h, p)| *h = *h - 2_f32 * p / magnitude_squared);
    householder
}

fn outer_product(x: Vec<f32>) -> Vec<f32> {
    // returns a a symetric matrix of length x length
    let length = x.len();
    assert!(length > 0, "needs to have non-zero length");
    let mut new_data = vec![0_f32; length * length];
    for i in 0..length {
        for j in 0..length {
            new_data[i * length + j] = x[i] * x[j];
        }
    }
    new_data
}

fn main() {
    let mut data = vec![0_f32; 3];
    let mut dims = vec![0; 2];
    dims[0] = 3;
    dims[1] = 1;
    data[0] = 0_f32;
    data[1] = 4_f32;
    data[2] = 1_f32;
    let x = blas::NdArray::new(dims, data.clone());
    // let fact = proto_householder(&ndarray);
    let h = hoseholder_matrix(&data);
    println!("Cross Product: {:?}", h);

    let test = blas::tensor_mult(1, &h, &x);
    println!("Projection: {:?}", test);
}

// fn main () {
//     let mut data = vec![0_f32;16];
//     let dims = vec![4;2];
//     for i in 0..8 {
//         data[i] = i as f32 + 1_f32;
//     }
//     data[8] = 1_f32;
//     data[9] = 1_f32;
//     data[10] = 3_f32;
//     data[11] = 3_f32;
//     data[12] = 2_f32;
//     data[13] = 1_f32;
//     data[14] = 1_f32;
//     data[15] = 1_f32;
//     let ndarray = blas::NdArray::new(dims, data);
//     // println!("test {:?}", ndarray);

//     let fact = lu_factorization(&ndarray);
//     println!("Lower: {:?}", fact.0);
//     println!("Upper: {:?}", fact.1);
// }
