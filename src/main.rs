#![allow(warnings)]
mod math;
mod blas;
use rayon::prelude::ParallelIterator;
use rayon::prelude::*;
use blas::{NdArray, Matrix};

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

fn proto_tensor_mult(blocksize:usize, x:NdArray, y:NdArray) -> NdArray {
    assert!(blocksize > 0);
    assert_eq!(x.dims[1], y.dims[0], "dimension mismatch");
    let mut value: f32;
    let x_rows = x.dims[0];
    let x_cols = x.dims[1];
    let y_rows = y.dims[0];
    let y_cols = y.dims[1];
    let mut new:Vec<f32> = vec![0_f32; x_rows * y_cols];
    for i in (0..x_rows).step_by(blocksize) {
        for j in (0..y_cols).step_by(blocksize) {
            for k in 0..(x_cols + blocksize - 1) / blocksize{
                for ii in 0..blocksize.min(x_rows - i) {
                    for jj in 0..blocksize.min(y_cols -j){
                        for kk in 0..blocksize.min(x_cols - k * blocksize) {
                            let index = (i + ii) * y_cols + jj + j;
                            let x_index = (i + ii ) * x_cols + k * blocksize + kk;
                            let y_index =  (k * blocksize + kk) * y_cols + jj + j;
                            new[index] +={
                                x.data[x_index]
                                * y.data[y_index]
                            };
                        }
                    }
                }
            }
        }
    };
    let mut dims = x.dims.clone();
    dims[1] = y.dims[1];
    NdArray::new ( dims, new )
}

use rand::Rng;

fn generate_matrix(rows: usize, cols: usize) -> NdArray {
    let mut rng = rand::thread_rng();
    let data: Vec<f32> = (0..rows * cols).map(|_| rng.gen_range(-10.0..10.0)).collect();
    NdArray::new(vec![rows, cols], data)
}

fn main() {
    // let i = 3;  // Number of rows in x
    // let j = 4;  // Number of columns in x, also rows in y
    // let k = 5;  // Number of columns in y

    // let x = generate_matrix(i, j);
    // let y = generate_matrix(j, k);

    // let blocksize = 4; // Test with different block sizes
    // let result = proto_tensor_mult(blocksize, x, y);


    let x = NdArray::new(vec![2, 3], vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    ]);

    let y = NdArray::new(vec![3, 2], vec![
        7.0, 8.0,
        9.0, 10.0,
        11.0, 12.0,
    ]);

    let result = proto_tensor_mult(2, x, y);
    println!("{:?}", result);


    // println!("{:?}", result);
}




// fn main () {
//     let x_dim = 3;
//     let y_dim = 3;

//     let test = usize::pow(x_dim, 2);
    
//     let mut x:Vec<f32> = Vec::with_capacity(usize::pow(x_dim, 2));
//     let mut y:Vec<f32> = Vec::with_capacity(usize::pow(y_dim, 2));
//     let mut dims:Vec<usize> = Vec::with_capacity(2);

//     let mut j = 0;
//     for i in 0..usize::pow(x_dim, 2) {
//         j = i / x_dim;
//         if (i - j) % x_dim == 0 {
//             x.push(1_f32)
//         } else {
//             x.push(0_f32);
//         }
//     }
//     for i in 0..usize::pow(y_dim, 2) {
//         y.push(i as f32)
//     }

//     for _ in 0..2 {
//         dims.push(x_dim)
//     }

//     let x_array = NdArray::new(dims.clone(), x);
//     let y_array = NdArray::new(dims.clone(), y);

//     println!("X array: {:?}", x_array);
//     println!("y array: {:?}", y_array);
//     let result = proto_tensor_mult(4, x_array, y_array);
//     println!("Output result\n {:?}", result); 

//     // let A = Matrix::new(4, 1, others);
//     // let transpose = transpose(A);
//     // println!("Transpose: {:?}", transpose);
// }
