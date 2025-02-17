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

// [[1,2,3,4] X 4] * [[3,4, 5,6] X 4] = [[30, 40, 50, 60]X4]



fn proto_tensor_mult(blocksize:usize, x:NdArray, y:NdArray) -> NdArray {
    assert!(blocksize > 0);
    assert_eq!(x.dims[0], y.dims[1], "dimension mismatch");
    let blocksize:usize = 2;
    // fixed matrix size ~ M[4, 4] for prototyping
    let mut value: f32;
    let x_rows = x.dims[0];
    let x_cols = x.dims[1];
    let y_rows = y.dims[0];
    let y_cols = y.dims[1];
    let mut new:Vec<f32> = vec![0_f32; x_rows * y_cols];
    for i in (0..x_rows).step_by(blocksize) {
        for j in (0..y_cols).step_by(blocksize) {
            for k in 0..(y_cols + 1) / blocksize{
                for ii in 0..blocksize - (i + blocksize) % x_rows % blocksize {
                    for jj in 0..blocksize - (j + blocksize) % y_cols % blocksize {
                        for kk in 0..blocksize {
                            if k * blocksize + kk  >= x_rows {
                            } else {
                                let index = (i + ii) * y_rows + jj + j;
                                let x_index = (i + ii ) * x_rows + k * blocksize + kk;
                                let y_index =  (k * blocksize + kk) * y_rows + jj + j;

                                let value ={
                                    x.data[x_index]
                                    * y.data[y_index]
                                };
                                new[index] += value;
                            }
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

fn main () {
    let x_dim = 5;
    let y_dim = 5;

    let test = usize::pow(x_dim, 2);
    
    let mut x:Vec<f32> = Vec::with_capacity(usize::pow(x_dim, 2));
    let mut y:Vec<f32> = Vec::with_capacity(usize::pow(y_dim, 2));
    let mut dims:Vec<usize> = Vec::with_capacity(2);

    let mut j = 0;
    for i in 0..usize::pow(x_dim, 2) {
        j = i / x_dim;
        if (i - j) % x_dim == 0 {
            x.push(1_f32)
        } else {
            x.push(0_f32);
        }
    }
    for i in 0..usize::pow(y_dim, 2) {
        y.push(i as f32)
    }

    for _ in 0..2 {
        dims.push(x_dim)
    }

    let x_array = NdArray::new(dims.clone(), x);
    let y_array = NdArray::new(dims.clone(), y);

    println!("X array: {:?}", x_array);
    println!("y array: {:?}", y_array);
    let result = proto_tensor_mult(2, x_array, y_array);
    println!("Output result\n {:?}", result); 

    // let A = Matrix::new(4, 1, others);
    // let transpose = transpose(A);
    // println!("Transpose: {:?}", transpose);
}
