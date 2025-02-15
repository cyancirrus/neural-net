#![allow(warnings)]
mod math;
use rayon::prelude::ParallelIterator;
use rayon::prelude::*;

// struct MatrixOperations {
// }

struct Matrix {
    rows:usize,
    cols:usize,
    data:Vec<f32>,
}
    
impl Matrix {
    fn new(rows:usize, cols:usize, data:Vec<f32>) -> Matrix {
        assert!(rows >= 0, "rows is not greater than or equal to 0");
        assert!(cols >= 0, "cols is not greater than or equal to 0");
        assert_eq!(data.len(), rows * cols, "dimension mismatch in matrix");
        Matrix { rows, cols, data }
    }
}

#[allow(non_snake_case)]
fn transpose_optimized(X:Matrix) -> Vec<f32> {
    let length:usize = X.rows * X.cols;
    let mut index:usize;
    (0..length).collect::<Vec<usize>>().par_iter()
        .map(|i| X.data[i * X.cols % length + i / X.rows])
        .collect::<Vec<f32>>()
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




#[allow(non_snake_case)]
fn column_iterator(rows:usize, cols:usize) -> Vec<usize> {
    let length:usize = rows * cols;
    let mut index:usize;
    (0..length).collect::<Vec<usize>>().par_iter()
        .map(|i| i * cols % length + i / rows)
        .collect::<Vec<usize>>()
}

// 3 x 2 * 2 x 8
fn matrix_multiplication(left:Matrix, right:Matrix) -> Matrix {
    let mut new:Vec<f32> = vec![0f32;left.rows * right.cols];
    let index:usize=0;
    let mut left_slice:&[f32];
    let mut right_slice:&[f32];
    let mut right_enum = right.data.into_iter().enumerate();
    for i in 0..left.rows {
        left_slice = &left.data[i*left.cols..(i+1)*left.cols];
        for j in 0..right.cols {
            new[i] = right_enum.clone()
                .filter(|(index, value)| index % left.rows == i)
                .map(|(_, value)| value)
                .zip(left_slice)
                .map(|(a_ij, b_ji)| a_ij * b_ji)
                .sum();
        }
    }
    Matrix::new( left.rows, right.cols, new )
}

// matrix 2 x 2 ~ 0..4
// 0, 1,
// 2, 3,

// 0 = (i=0 * col)=0  + i % cols)
// 2 = (i=1 * col)=2 
// 1 = (i=2 * col)=6 + (i + 1 )% cols ~ 7
// 3 = (i=3 * col)=8 ~ i % cols ~ 9


// matrix 2 x 3 ~ 0..6
// 0 = (i=0 * col)=0  + i= / cols)
// 2 = (i=1 * col)=2 
// 4 = (i=2 * col)=4 
// 1 = (i=3 * col)=6 + i % cols ~ 7
// 3 = (i=4 * col)=8 ~ i % cols ~ 9
// 5 = (i=5 * col)=10 ~ i % cols ~ 11

// matrix 2 x 2 ~ 0..4
// 0, 1,
// 2, 3,


fn main () {
    let mut numbers:Vec<f32> = Vec::with_capacity(4);
    let mut others:Vec<f32> = Vec::with_capacity(4);

    for i in 0..4 {
        numbers.push(i as f32);
        others.push(i as f32);
    }
    let A = Matrix::new(4, 1, others);
    let transpose = transpose(A);
    println!("Transpose: {:?}", transpose);
    // // let transpose = transpose(3, 2, &others);
    // let column_iter = column_iterator(3, 2);
    // println!("Matrix: {:?}", numbers);
    // println!("Column Iterator: {:?}", column_iter);
}
