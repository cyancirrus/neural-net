#![allow(warnings)]
use std::cmp::min;
use rayon::prelude::{ParallelIterator};
use rayon::prelude::*;


pub fn dot_product(x:&[f32], y:&[f32]) -> f32 {
    x.iter().zip(y.iter()).map(|(&x, &y)| x * y).sum()
}

pub fn sigmoid(x:f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn identity(x:f32) -> f32 {
    x
}

pub fn square_matrix_transpose(mut matrix:Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let rows = matrix.len();
    let cols= matrix[0].len();
    for row in (0..rows) {
        for col in (0..cols) {
            if row < cols {
                let aij = matrix[row][col];
                let aji = matrix[col][row];
                matrix[row][col] = aji;
                matrix[col][row] = aij;
            }
        }
    }
    matrix
}

pub fn matrix_transpose(matrix:&Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut transpose = vec![vec![0_f32;rows];cols];
    for row in 0..rows {
        for col in 0..cols {
            transpose[col][row] = matrix[row][col];
        }
    }
    transpose
}

pub fn vector_diff(x:&[f32], y:&[f32]) -> Vec<f32> {
    x.iter().zip(y.iter()).map(|(&x, &y)| x - y).collect()
}

pub fn vector_add(x:&[f32], y:&[f32]) -> Vec<f32> {
    x.iter().zip(y.iter()).map(|(&x, &y)| x + y).collect()
}

pub fn vector_product(x:&[f32], y:&[f32]) -> Vec<f32> {
    x.iter().zip(y.iter()).map(|(&x, &y)| x * y).collect()
}

pub fn scalar_product(lambda:f32, vector:&[f32]) -> Vec<f32> {
    vector.iter().map(|&vector| lambda * vector).collect()
}
    
pub fn loss_squared(prediction:Vec<f32>, result:Vec<f32>) -> f32 {
    let loss = prediction.par_iter().zip(result.par_iter())
        .map(|(p, r)| (p - r) * (p - r))
        .sum();
    loss
}
