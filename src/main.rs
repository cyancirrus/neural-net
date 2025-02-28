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

// fn householder_matrix(mut x: &[f32]) -> NdArray {
fn householder_params(mut x: &[f32]) -> (f32, Vec<f32>) {
    let length = x.len();
    assert!(length > 0, "needs to have non-zero length");
    println!("X: {:?}", x);
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
        householder.data[i * length + i] = 1_f32;
    }
    (2_f32 / magnitude_squared, u)
}

fn householder_factor_ugly(mut x: NdArray) -> NdArray {
    let rows = x.dims[0];
    let cols = x.dims[1];
    
    for o in 0..cols.min(rows) {
        let column_vector = (o..rows).into_par_iter().map(|r| x.data[r*cols + o]).collect::<Vec<f32>>();
        let (b, u) = householder_params(&column_vector);
        println!("Column vector: {:?}", column_vector);
        println!("Householder vector: {:?}, row: {}", u, o);
        let lrows = cols - o;
        let lcols = rows - o;
        let mut queue: Vec<(usize, f32)> = vec![(0, 0_f32); lcols * lrows];
        for i in o..rows {
            for j in o..cols {
                {
                    (o..rows).for_each(|k| {
                        queue[(i - o)*lcols + j - o].0 = i* cols + j;
                        queue[(i - o)*lcols + j - o].1 -= x.data[k*cols + j] * b * u[i-o] * u[k - o];
                    });
                }
            }
        }
        queue.iter().for_each(|q| x.data[q.0] += q.1);
        println!("{}th change: {:?}", o+1, x);
    }
    x
}

fn householder_factor(mut x: NdArray) -> NdArray {
    let rows = x.dims[0];
    let cols = x.dims[1];
    
    for o in 0..cols.min(rows) {
        let column_vector = (o..rows).into_par_iter().map(|r| x.data[r*cols + o]).collect::<Vec<f32>>();
        let (b, u) = householder_params(&column_vector);
        println!("Column vector: {:?}", column_vector);
        println!("Householder vector: {:?}, row: {}", u, o);
        let mut queue: Vec<(usize, f32)> = vec![(0, 0_f32); (cols - o) * (rows -o)];
        for i in 0..rows-o {
            for j in 0..cols-o{
                {
                if i <= j || i > o {
                    (0..rows-o).for_each(|k| {
                        queue[i*(cols - o) + j].0 = (i + o)* cols + (j+ o);
                        queue[i*(cols - o) + j].1 -= x.data[(k + o)*cols + (j + o)] * b * u[i] * u[k];
                        });
                    }
                }
            }
        }
        queue.iter().for_each(|q| x.data[q.0] += q.1);
        println!("{}th change: {:?}", o+1, x);
        for i in o+1..rows {
            println!("target ({}, {})", i + 1, o + 1);
            x.data[i*cols + o] = 0_f32;
        }
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
    data[1] = 1_f32;
    data[2] = 1_f32;
    data[3] = 1_f32;
    data[4] = 2_f32;
    data[5] = 3_f32;
    data[6] = 1_f32;
    data[7] = 1_f32;
    data[8] = 1_f32;
    // data[0] = 0_f32;
    // data[1] = 1_f32;
    // data[2] = 1_f32;
    // data[3] = 4_f32;
    // data[4] = 2_f32;
    // data[5] = 3_f32;
    // data[6] = 1_f32;
    // data[7] = 1_f32;
    // data[8] = 1_f32;
    let x = blas::NdArray::new(dims, data.clone());
    println!("input matrix {:?}", x);
    // let fact = proto_householder(&ndarray);
    let h = householder_factor(x);
    println!("hoseholder factor: {:?}", h);

    // let test = blas::tensor_mult(1, &h, &x);
    // println!("Projection: {:?}", test);
}
