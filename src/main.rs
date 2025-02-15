#![allow(warnings)]
mod math;
mod blas;
use rayon::prelude::ParallelIterator;
use rayon::prelude::*;
use blas::{Matrix};

struct NdArray {
    dims:Vec<usize>,
    data:Vec<f32>,
}


impl NdArray {
    pub fn new(dims:Vec<usize>, data:Vec<f32>) -> NdArray {
        NdArray { dims, data }
    }
}

#[allow(non_snake_case)]
fn transpose(X:Matrix) -> Vec<f32> {
    let mut new: Vec<f32> = vec![0_f32;X.rows*X.cols];
    for i in 0.. X.rows {
        for j in 0..X.cols {
            new[i * X.cols + j] = X.data[j * X.rows + i];
        }
    }
    new
}

#[allow(non_snake_case)]
fn matrix_multiplication(Left:Matrix, Right:Matrix) -> Matrix {
    assert_eq!(Left.cols, Right.rows, "dimensions do not match in matrix mult");
    let mut new:Vec<f32> = vec![0f32;Left.rows * Right.cols];
    let mut accum:f32 = 0f32;

    for i in 0..Left.rows {
        for j in 0..Right.cols {
            // common index between Left and Right
            for k in 0..Left.cols {
                accum += Left.data[i * Left.rows + k] * Right.data[j * Right.cols + k]
            }
        new[i * Left.rows + j * Right.cols] = accum;
        accum = 0_f32;
        }            
    }
    Matrix::new( Left.rows, Right.cols, new )
}

// [[1,2,3,4] X 4] * [[3,4, 5,6] X 4] = [[30, 40, 50, 60]X4]



// fn tensor_mult(blocksize:usize, x:NdArray, y:NdArray) -> NdArray {
//     assert!(blocksize > 0);
//     assert_eq!(x.dims[0], y.dims[1], "dimension mismatch");
//     let blocksize:usize = 2;
//     // fixed matrix size ~ M[4, 4] for prototyping
//     let mut new:Vec<f32> = vec![0_f32; 4 * 4];
//     let mut value: f32;
//     let x_rows = 4;
//     let x_cols = 4;
//     let y_rows = 4;
//     let y_cols = 4;

//     // iterate by blocksize
//     for i in (0..4).step_by(blocksize) {
//         for j in (0..4).step_by(blocksize) {
//             for k in 0..blocksize {
//                 for ii in 0..blocksize {
//                     for jj in 0..blocksize {
//                         for kk in 0..blocksize + x_cols % blocksize {
//                             let index = (i + ii) * x_rows +  k * blocksize + kk;
//                             let x_index = (i + ii ) * x_rows + j + jj;
//                             let y_index =  (j + jj) * y_cols +  k * blocksize + kk;
//                             let value ={
//                                 x.data[x_index]
//                                 * y.data[y_index]
//                             };
//                             if index == 4 {
//                                 println!("x index: {}, x value: {}", x_index, x.data[x_index]);
//                                 println!("y index: {}, y_value: {}", y_index, y.data[y_index]);
//                                 println!("index: {}, value: {}, i: {}, ii: {}, j:{}, jj:{}, k:{}, kk:{}, ", index, value, i, ii, j, jj, k, kk, );
//                             };
//                             new[index] += value;
//                         }
//                     }
//                 }
//             }
//         }
//     }
//     // Need to do like x.dim[1] = y.dim[x] and clone from x
//     NdArray::new ( x.dims, new )
// }

use rayon::prelude::*;

fn tensor_mult(blocksize: usize, x: NdArray, y: NdArray) -> NdArray {
    assert!(blocksize > 0);
    assert_eq!(x.dims[1], y.dims[0], "Dimension mismatch for matrix multiplication");

    let x_rows = x.dims[0];
    let x_cols = x.dims[1];
    let y_cols = y.dims[1];

    let mut new = vec![0_f32; x_rows * y_cols];

    // Parallelize over i and j to distribute work over thread pool
    new.par_iter_mut().enumerate().for_each(|(output_index, new_val)| {
        let i = output_index / y_cols; // Row index
        let j = output_index % y_cols; // Column index

        // Compute the dot product of row `i` from x and column `j` from y
        let mut sum = 0.0;
        for k in 0..x_cols {
            let x_index = i * x_cols + k;
            let y_index = k * y_cols + j;
            sum += x.data[x_index] * y.data[y_index];
        }

        *new_val = sum; // Store result in the correct index
    });

    NdArray::new(vec![x_rows, y_cols], new)
}



fn main () {
    let mut x:Vec<f32> = Vec::with_capacity(16);
    let mut y:Vec<f32> = Vec::with_capacity(16);
    let mut dims:Vec<usize> = Vec::with_capacity(2);
    
    let mut j = 0;
    for i in 0..16 {
        // x.push(i as f32);
        j = i / 4;
        if (i - j) % 4 == 0 {
            x.push(1_f32)
        } else {
            x.push(0_f32);
        }
    }
    // println!("Vector: {:?}", x);
    for i in 0..16 {
        y.push(i as f32)
    }

    for _ in 0..2 {
        dims.push(4)
    }

    let x_array = NdArray::new(dims.clone(), x);
    let y_array = NdArray::new(dims.clone(), y);


    let result = tensor_mult(2, x_array, y_array);
    println!("Output result: {:?}", result.data); 

    // let A = Matrix::new(4, 1, others);
    // let transpose = transpose(A);
    // println!("Transpose: {:?}", transpose);
}
