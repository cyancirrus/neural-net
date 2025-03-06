#![allow(warnings)]
use blas::{Matrix, NdArray};
use neural_net::calc_utils::blas;
use neural_net::calc_utils::math;
use neural_net::calc_utils::simd;
use neural_net::calc_utils::qr_decomposition;
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
    triangle: NdArray,
}

struct SchurDecomp {
    rotation: NdArray, // The accumulated orthogonal transformations (U for SVD)
    kernel: NdArray, // The upper quasi-triangular matrix (Schur form)
}

impl SchurDecomp {
    pub fn new(rotation:NdArray, kernel:NdArray) -> Self {
        Self { rotation, kernel }
    }
}


impl QrDecomposition {
    fn new(projections: Vec<HouseholderReflection>, triangle:NdArray) -> Self{
        Self { projections, triangle }
    }
    fn left_multiply(&self, other:NdArray) -> NdArray {
        let dims = self.triangle.dims.clone();
        // let mut data = vec![0_f32; dims[0] * dims[1]];
        let mut data = other.data.clone();
        let rows = other.dims[0];

        (0..other.dims[0]).for_each(|i| {
            let start = i * rows;
            let end = (i  + 1) * rows;
            let row = &data[start..end];
            let cordinate = self.determine_basis(row.to_vec());
            for k in 0..cordinate.len() {
                // If you want to generate columns
                // data[k * dims[0] + i] = cordinate[k];
                data[i * dims[0] + k] = cordinate[k];
            }
        });
        NdArray::new(dims, data)
    }
    // fn right_multiply(&self, ndarray:NdArray) -> NdArray {
    //     let dims = self.triangle.dims.clone();
    //     // let mut data = vec![0_f32; dims[0] * dims[1]];
    //     let mut data = ndarray.data.clone();
    //     let rows = ndarray.dims[0];

    //     // for i in 0..dims[1] {
    //     (0..ndarray.dims[0]).rev().for_each(|i| {
    //         let start = i * rows;
    //         let end = (i  + 1) * rows;
    //         let row = &data[start..end];
    //         let cordinate = self.determine_basis(row.to_vec());
    //         for k in 0..cordinate.len() {
    //             // If you want to generate columns
    //             // data[k * dims[0] + i] = cordinate[k];
    //             data[i * dims[0] + k] = cordinate[k];
    //         }
    //     });
    //     NdArray::new(dims, data)
    // }
    fn triangle_rotation(&self) -> NdArray {
        let dims = self.triangle.dims.clone();
        let mut data = self.triangle.data.clone();
        let rows = self.triangle.dims[0];

        (0..self.triangle.dims[0]).rev().for_each(|i| {
            let start = i * rows;
            let end = (i  + 1) * rows;
            let mut row = &data[start..end];
            let cordinate = self.determine_basis(row.to_vec());
            for k in 0..cordinate.len() {
                data[i * dims[0] + k] = cordinate[k];
            }
        });
        NdArray::new(dims, data)
    }
    

    fn determine_basis(&self, mut data:Vec<f32>) -> Vec<f32> {
        let mut queue = vec![0_f32; self.triangle.dims[0]];
        let mut delta = vec![0_f32; self.triangle.dims[0]];
        for i in (0..self.projections.len()) {
            let mut delta = vec![0_f32; self.triangle.dims[0]];
            let projection = &self.projections[i];
            for j in 0..projection.vector.len() {
                for  k in 0..projection.vector.len() {
                    delta[i + j] -= projection.beta *  projection.vector[k] * projection.vector[j] * data[i + k];
                }
            }
            for j in 0..delta.len() {
                data[j] += delta[j];
            }
    }
    data
    }
}

fn givens_rotation(ndarray:&NdArray) -> NdArray {
    assert_eq!(ndarray.data.len(), 2);
    let mut givens = vec![0_f32; 4]; 

    let t:f32;
    let s:f32;
    let c:f32;
    
    if ndarray.data[1].abs() > ndarray.data[0].abs() {
        t = ndarray.data[0]/ndarray.data[1];
        s = 1_f32/(1_f32 + t.powi(2)).sqrt();
        c = s*t;
    } else {
        t = ndarray.data[1]/ndarray.data[0];
        c = 1_f32/(1_f32 + t.powi(2)).sqrt();
        s = c*t;
    }
    givens[0]=c;
    givens[1]=s;
    givens[2]=-s;
    givens[3]=c;
    NdArray::new(vec![2;2], givens)
}


fn householder_params(mut x: &[f32]) -> HouseholderReflection {
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
        householder.data[i * length + i] = 1_f32;
    }
    HouseholderReflection::new(2_f32 / magnitude_squared, u)
}

fn qr_decompose(mut x: NdArray) -> QrDecomposition {
    let rows = x.dims[0];
    let cols = x.dims[1];
    let mut projections = Vec::with_capacity(cols.min(rows));
    
    for o in 0..cols.min(rows) - 1 {
        let column_vector = (o..rows).into_par_iter().map(|r| x.data[r*cols + o]).collect::<Vec<f32>>();
        let householder = householder_params(&column_vector);
        projections.push(householder);
        let mut queue: Vec<(usize, f32)> = vec![(0, 0_f32); (cols - o)  * (rows -o)];
        for i in 0..(rows-o).min(cols-o) {
            for j in 0..cols-o {
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

fn real_schur_iteration(mut schur:SchurDecomp) -> SchurDecomp {
    let rows = schur.kernel.dims[0];
    let mut qr = qr_decompose(schur.kernel);
    let rotation = qr.left_multiply(schur.rotation);  // RQ = Q'AQ
    let kernel = qr.triangle_rotation();
    SchurDecomp { rotation, kernel }
}

fn real_schur_threshold(kernel:&NdArray) -> f32 {
    let rows = kernel.dims[0];
    let cols = kernel.dims[1];
    let mut off_diagonal = 0_f32;

    for j in 0..cols {
        for i in j+1..rows {
            off_diagonal += kernel.data[i * rows + j].abs();

        }
    }
    off_diagonal
}

fn real_schur_decomp(mut kernel:NdArray) -> SchurDecomp {
    let STOP_CONDITION:f32 = 1e-6;
    let rows = kernel.dims[0];
    let mut identity = blas::create_identity_matrix(rows);
    let mut schur = SchurDecomp::new(identity, kernel);

    while real_schur_threshold(&schur.kernel) > STOP_CONDITION  {
        schur = real_schur_iteration(schur);
    }
    schur
}


fn main() {
    let mut data = vec![0_f32; 9];
    let mut dims = vec![2; 2];
    data[0] = 1_f32;
    data[1] = 2_f32;
    data[2] = 3_f32;
    data[3] = 4_f32;
    let x = blas::NdArray::new(dims, data.clone());
    println!("x: {:?}", x);
    // let qr = qr_decompose(x);
    // println!("qr {:?}", qr.ndarray);
    let real_schur = real_schur_decomp(x);
    println!("real schur {:?}", real_schur.kernel);
    // println!("real schur {:?}", real_schur);

    let mut g_data = vec![0_f32;2];
    g_data[0] = 1_f32;
    g_data[1] = 1_f32/2_f32;
    let mut g_dims = vec![0;2];
    g_dims[0]=2;
    g_dims[1]=1;

    let g_input = NdArray::new(g_dims, g_data);
    
    let givens = givens_rotation(&g_input);
    println!("Givens rotation {:?}", givens);

    let rotate_vec = blas::tensor_mult(1, &givens, &g_input);
    println!("Rotated Vec {:?}", rotate_vec);

}

// fn main() {
//     let mut data = vec![0_f32; 9];
//     let mut dims = vec![0; 2];
//     dims[0] = 3;
//     dims[1] = 3;
//     // data[0] = 0_f32;
//     // data[1] = 1_f32;
//     // data[2] = 1_f32;
//     // data[3] = 1_f32;
//     // data[4] = 2_f32;
//     // data[5] = 3_f32;
//     // data[6] = 1_f32;
//     // data[7] = 1_f32;
//     // data[8] = 1_f32;
//     data[0] = 9_f32;
//     data[1] = 8_f32;
//     data[2] = -7_f32;
//     data[3] = 6_f32;
//     data[4] = 5_f32;
//     data[5] = 4_f32;
//     data[6] = 3_f32;
//     data[7] = 2_f32;
//     data[8] = -1_f32;
//     let x = blas::NdArray::new(dims, data.clone());
//     // println!("input matrix {:?}", x);
//     // let h = qr_decompose(x);
//     // println!("hoseholder factor: {:?}", h.ndarray);

//     // // let orth = h.retrieve_q();
//     // let identity = blas::create_identity_matrix(3);
//     // println!("Identity matrix {:?}", identity);
//     // let orth = h.q_multiply(identity);
//     // println!("orthogonal ~= {:?}", orth);

//     // let retrieve = blas::tensor_mult(3, &orth, &h.ndarray);
//     // println!("check work {:?}", retrieve);

//     let real_schur = real_schur_decomp(x);
//     println!("real schur {:?}", real_schur);
//     // let test = blas::tensor_mult(1, &h, &x);
//     // println!("Projection: {:?}", test);
// }
