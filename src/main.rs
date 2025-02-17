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
    let mut counter:usize = 0;
    let mut all_counter:usize = 0;
    let mut inner_counter:usize = 0;
    for i in (0..x_rows).step_by(blocksize) {
        for j in (0..y_cols).step_by(blocksize) {
            for k in 0..(y_cols + 1) / blocksize{
                for ii in 0..blocksize {
                    for jj in 0..blocksize {
                        for kk in 0..blocksize {
                            all_counter += 1;
                            if  i + ii  >= x_rows {
                            } else if j + jj >=  y_cols {
                            } else if k * blocksize + kk  >= x_rows {
                            } else {
                                inner_counter+=1;

                                let index = (i + ii) * y_rows + jj + j;
                                let x_index = (i + ii ) * x_rows + k * blocksize + kk;
                                let y_index =  (k * blocksize + kk) * y_rows + jj + j;
                                
                                let hindex = (i + ii) * y_rows + jj + j;
                                let hx_index = (i + ii ) * x_rows + k * blocksize + kk;
                                let hy_index =  (k * blocksize + kk) * y_rows + jj + j;
                                 
                                if (index==6) {
                                    println!("Hypothesized: ( {}, {} )", (hindex / x_rows) + 1, (hindex % x_rows) + 1);
                                    println!("x point: ( {}, {} )", (hx_index / x_rows) + 1, (hx_index % x_rows) + 1);
                                    println!("y point: ( {}, {} )", (hy_index / y_rows) + 1, (hy_index % y_rows) + 1);
                                    println!("index: {},  i: {}, ii: {}, j:{}, jj:{}, k:{}, kk:{}, ", index, i, ii, j, jj, k, kk, );
                                };

                                let value ={
                                    x.data[x_index]
                                    * y.data[y_index]
                                };
                                new[index] += value;
                                counter +=1;
                            }
                        }
                    }
                }
            }
        }
    };
    println!("Total Iter: {}, Available Iter: {}, Valid Iter: {}", all_counter, inner_counter, counter);
    let mut dims = x.dims.clone();
    dims[1] = y.dims[1];
    NdArray::new ( dims, new )
}

fn main () {
    let x_dim = 4;
    let y_dim = 4;

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

// fn proto_tensor_mult(blocksize:usize, x:NdArray, y:NdArray) -> NdArray {
//     assert!(blocksize > 0);
//     assert_eq!(x.dims[0], y.dims[1], "dimension mismatch");
//     let blocksize:usize = 2;
//     // fixed matrix size ~ M[4, 4] for prototyping
//     let mut value: f32;
//     let x_rows = x.dims[0];
//     let x_cols = x.dims[1];
//     let y_rows = y.dims[0];
//     let y_cols = y.dims[1];
//     let mut new:Vec<f32> = vec![0_f32; x_rows * y_cols];
//     let mut counter:usize = 0;
//     for i in (0..x_rows).step_by(blocksize) {
//         for j in (0..y_cols).step_by(blocksize) {
//             for k in 0..(y_cols + 1) / blocksize{
//                 let i_x = blocksize - (i + blocksize) % x_rows % blocksize;
//                 let j_x = blocksize - (j + blocksize) % y_cols % blocksize;
//                 let k_x = i_x.min(j_x);
//                 for ii in 0..blocksize - (i + blocksize) % x_rows % blocksize {
//                     for jj in 0..blocksize - (j + blocksize) % y_cols % blocksize {
//                         // println!("k_x:{}, i:{}, ii:{}, i_x:{}, j:{}, jj:{}, j_x:{}", k_x, i, ii, i_x, j, jj, j_x,);
//                         // for kk in 0..i_x.min(j_x) {
//                         for kk in 0..i_x.min(j_x) {
//                             let index = (i + ii) * x_rows +  k * blocksize + kk;
//                             let x_index = (i + ii ) * x_rows + j + jj;
//                             let y_index =  (j + jj) * y_cols +  k * blocksize + kk;
//                             let value ={
//                                 x.data[x_index]
//                                 * y.data[y_index]
//                             };
//                             // if (j == 0) & (jj == 0) & (kk == 0) {
//                             // if (i == 0) & (ii == 0) & (x.data[x_index] == 1_f32) {
//                             // if (x_index == 0) & (x_index == 4) {
//                                 println!("x index: {}, x value: {}", x_index, x.data[x_index]);
//                                 println!("y index: {}, y value: {}", y_index, y.data[y_index]);
//                                 // println!("index: {}, value: {}, i: {}, ii: {}, j:{}, jj:{}, k:{}, kk:{}, ", index, value, i, ii, j, jj, k, kk, );
//                             // };
//                             // println!("Value: {}", value);
//                             new[index] += value;
//                             // println!("out kk/in jj");
//                         }
//                         // println!("out jj/in ii");
//                     }
//                 }
//             }
//         }
//     };
//     let mut dims = x.dims.clone();
//     dims[1] = y.dims[1];
//     NdArray::new ( dims, new )
// }
