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

struct HouseholderReflection {
    beta:f32, // store 2 / u'u
    vector:Vec<f32>, // stores reflection u
}

impl HouseholderReflection {
    fn new(beta:f32, vector:Vec<f32>) -> Self {
        Self { beta, vector }
    }
}

struct QrDecomposition {
    projections: Vec<HouseholderReflection>,
    ndarray: NdArray,
}

impl QrDecomposition {
    fn new(projections: Vec<HouseholderReflection>, ndarray:NdArray) -> Self{
        Self { projections, ndarray }
    }

    fn retrieve_q(&self) -> NdArray {
        let dims = self.ndarray.dims.clone();
        let mut data = vec![0_f32; dims[0] * dims[1]];

        // for i in 0..dims[1] {
        for i in 0..dims[0] {
            let cordinate = self.determine_basis(i);
            println!("cordinate appears as {:?}", cordinate);
            for k in 0..cordinate.len() {
                data[k * dims[0] + i] = cordinate[k];
            }
        }
        NdArray::new(dims, data)
    }
    fn determine_basis(&self, e:usize) -> Vec<f32> {
        assert!(e < self.ndarray.dims[0]);
        let mut data = vec![0_f32;self.ndarray.dims[0]];
        let mut queue = vec![0_f32; self.ndarray.dims[0]];
        data[e] = 1_f32;

        println!("data: {:?}", data);
        for i in 0..self.projections.len() {
            let projection = &self.projections[i];
            let mut delta = vec![0_f32; projection.vector.len()];
            for j in 0..projection.vector.len() {
                for  k in 0..projection.vector.len() {
                    delta[j] -= projection.beta *  projection.vector[k] * projection.vector[j] * data[i + k];
                }
            }
            for j in 0..delta.len() {
                data[i + j] += delta[j];
            }
            println!("post: {:?}", data);
        }
    data
    }
}


fn householder_params(mut x: &[f32]) -> HouseholderReflection {
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
    HouseholderReflection::new(2_f32 / magnitude_squared, u)
}

fn householder_factor(mut x: NdArray) -> QrDecomposition {
    let rows = x.dims[0];
    let cols = x.dims[1];
    let mut projections = Vec::with_capacity(cols.min(rows));
    
    for o in 0..cols.min(rows) {
        let column_vector = (o..rows).into_par_iter().map(|r| x.data[r*cols + o]).collect::<Vec<f32>>();
        let householder = householder_params(&column_vector);
        projections.push(householder);
        let mut queue: Vec<(usize, f32)> = vec![(0, 0_f32); (cols - o)  * (rows -o)];
        for i in 0..(rows-o).min(cols-o) {
            for j in 0..cols-o{
                // Need to compute the change for everything to the right of the initial vector
                if i <= j || j > o {
                    let sum = (0..rows-o).into_par_iter().map(|k| {
                        x.data[(k + o)*cols + (j + o)] * projections[o].beta * projections[o].vector[i] * projections[o].vector[k]
                    }).sum();
                    queue[i*(cols - o) + j].0 = (i + o)* cols + (j+ o);
                    queue[i*(cols - o) + j].1 = sum;
                }
            }
        }
        queue.iter().for_each(|q| x.data[q.0] -= q.1);
        (o+1..rows).for_each(|i| x.data[i*cols + o] = 0_f32);
    }
    QrDecomposition::new(projections, x)
}

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
    // data[1] = 1_f32; data[2] = 1_f32; data[3] = 4_f32; data[4] = 2_f32; data[5] = 3_f32; data[6] = 1_f32; data[7] = 1_f32;
    // data[8] = 1_f32;
    let x = blas::NdArray::new(dims, data.clone());
    println!("input matrix {:?}", x);
    let h = householder_factor(x);
    println!("hoseholder factor: {:?}", h.ndarray);

    let orth = h.retrieve_q();
    println!("orthogonal ~= {:?}", orth);

    let retrieve = blas::tensor_mult(3, &orth, &h.ndarray);
    println!("check work {:?}", retrieve);

    // let test = blas::tensor_mult(1, &h, &x);
    // println!("Projection: {:?}", test);
}
