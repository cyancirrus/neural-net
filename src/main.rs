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

// fn householder_matrix(mut x: &[f32]) -> NdArray {
fn householder_matrix(mut x: &[f32]) -> (f32, Vec<f32>) {
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
    // let projection = math::outer_product(u);
    // householder.data.par_iter_mut()
    //     .zip(projection.par_iter())
    //     .for_each(|(h, p)| *h = *h - 2_f32 * p / magnitude_squared);
    // householder
    (2_f32 / magnitude_squared, u)
}

fn hoseholder_factor(mut x: NdArray) -> NdArray {
    let rows = x.dims[0];
    let cols = x.dims[1];
    
    for j in 0..cols {
        let column_vector = (0..rows).into_par_iter().map(|r| x.data[r*cols + j]).collect::<Vec<f32>>();
        let (b, u) = householder_matrix(&column_vector);
        println!("householder vector: {:?}", u);
        x.data[j*cols + j] *= 1_f32 - u[0].powi(2)*b;
        for i in j..rows {
            for k in i..cols {
                if i == k {
                    x.data[i*cols + j] += x.data[k*cols + j] * (1_f32 - b *u[i].powi(2));
                } else {
                    x.data[i*cols + j] -= x.data[k*cols + j] * b * u[k] * u[i];
                    x.data[k*cols + j] = 0_f32;
                }
            }
        }
        println!("data: {:?}", x);

    }
    x
}


// fn main() {
//     let mut data = vec![0_f32; 3];
//     let mut dims = vec![0; 2];
//     dims[0] = 3;
//     dims[1] = 1;
//     data[0] = 0_f32;
//     data[1] = 4_f32;
//     data[2] = 1_f32;
//     let x = blas::NdArray::new(dims, data.clone());
//     // let fact = proto_householder(&ndarray);
//     let h = hoseholder_matrix(&data);
//     println!("Cross Product: {:?}", h);

//     let test = blas::tensor_mult(1, &h, &x);
//     println!("Projection: {:?}", test);
// }

fn main() {
    let mut data = vec![0_f32; 9];
    let mut dims = vec![0; 2];
    dims[0] = 3;
    dims[1] = 3;
    data[0] = 0_f32;
    data[1] = 4_f32;
    data[2] = 1_f32;
    data[3] = 4_f32;
    data[4] = 1_f32;
    data[5] = 1_f32;
    data[6] = 1_f32;
    data[7] = 1_f32;
    data[8] = 1_f32;
    let x = blas::NdArray::new(dims, data.clone());
    // let fact = proto_householder(&ndarray);
    let h = hoseholder_factor(x);
    println!("hoseholder factor: {:?}", h);

    // let test = blas::tensor_mult(1, &h, &x);
    // println!("Projection: {:?}", test);
}
