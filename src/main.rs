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

struct GivensDecomposition {
    kernel: NdArray,
    rotation: NdArray,
}

impl GivensDecomposition {
    pub fn new(kernel:NdArray, rotation: NdArray) -> Self {
        Self { kernel, rotation }
    }
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

fn givens_rotation(a:f32, b:f32) -> NdArray {
    let mut givens = vec![0_f32; 4]; 

    let t:f32;
    let s:f32;
    let c:f32;
    
    if b.abs() > a.abs() {
        t = a/b;
        s = 1_f32/(1_f32 + t.powi(2)).sqrt();
        c = s*t;
    } else {
        t = b/a;
        c = 1_f32/(1_f32 + t.powi(2)).sqrt();
        s = c*t;
    }
    givens[0]=c;
    givens[1]=s;
    givens[2]=-s;
    givens[3]=c;
    NdArray::new(vec![2;2], givens)
}

// fn jacobi_rotation(i:usize, j:usize, ndarray:&NdArray) -> NdArray {
//     let rows = ndarray.dims[0];
//     let cols = ndarray.dims[1];
//     let mut jacobi = vec![0_f32; 4];
//     let mut s = 0_f32;

//     s += ndarray.data[i*cols + j].powi(2) + ndarray.data[j*cols + i].powi(2);
//     let b = 2_f32 * ndarray.data[i* cols + j] / ( ndarray.data[i *cols + i] - ndarray.data[j * cols + j]);
//     let t = b.signum() / (b.abs() + (b.powi(2) + 1_f32).sqrt());
//     let c = 1_f32 / (t.powi(2) + 1_f32).sqrt();
//     let s = c*t;

//     jacobi[0]=c;
//     jacobi[1]=s;
//     jacobi[2]=-s;
//     jacobi[3]=c;
//     NdArray::new(vec![2;2], jacobi)
// }

fn jacobi_rotation(i:usize, j:usize, ndarray:&NdArray) -> NdArray {
    let rows = ndarray.dims[0];
    let cols = ndarray.dims[1];
    let mut jacobi = vec![0_f32; 4];
    let mut s = 0_f32;

    s += ndarray.data[i*cols + j].powi(2) + ndarray.data[j*cols + i].powi(2);
    let magnitude = ((ndarray.data[j*cols + j] + ndarray.data[i*cols + i]) / ndarray.data[j*cols + i]).powi(2);
    let s = 1_f32 / (magnitude + 1_f32).sqrt();
    let c = 1_f32 - s.powi(2);

    jacobi[0]=c;
    jacobi[1]=s;
    jacobi[2]=-s;
    jacobi[3]=c;
    NdArray::new(vec![2;2], jacobi)
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

fn transpose(mut ndarray: NdArray) -> NdArray {
    let rows = ndarray.dims[0];
    let cols = ndarray.dims[1];
    for i in 0..rows {
        for j in i+1..cols {
            let temp: f32 = ndarray.data[i*rows + j];
            ndarray.data[i*rows + j] = ndarray.data[j*rows + i];
            ndarray.data[j*rows + i] = temp;
        }
    }
    ndarray.dims[0] = cols;
    ndarray.dims[1] = rows;
    ndarray
}


fn givens_iteration(first:bool, mut givens:GivensDecomposition) -> GivensDecomposition {
    let rows = givens.kernel.dims[0];
    let cols = givens.kernel.dims[1];
    let rotation:NdArray;
    let left_rotation = givens_rotation(givens.kernel.data[0], givens.kernel.data[1]);
    // // if !first {
        givens.kernel = blas::tensor_mult(2, &left_rotation, &givens.kernel);
    //     println!("upper cancel {:?}", givens.kernel);
    // // }
    let right_rotation  = transpose(left_rotation.clone());
    // let right_rotation  = givens_rotation(givens.kernel.data[1], -givens.kernel.data[3]);
    givens.kernel = blas::tensor_mult(2, &givens.kernel, &right_rotation);
    println!("re-pivot {:?}", givens.kernel);
    let ortho_check = blas::tensor_mult(2, &left_rotation, &right_rotation);
    println!("Ortho check in givens {:?}", ortho_check);
    givens
}

fn givens_decomp(mut kernel:NdArray) -> GivensDecomposition {
    println!("Kernel {:?}", kernel);
    let mut upper = true;
    let rotation = blas::create_identity_matrix(kernel.dims[0]);
    let mut givens = GivensDecomposition { rotation, kernel };
    let mut iteration = 8;
    let mut first = true;
    while iteration > 0 {
        iteration -=1;
        upper = !upper;
        givens = givens_iteration(first, givens);
        println!("Givens iteration {:?}", givens.kernel);
        first = false;
    }
    givens
}


fn make_symmetric(kernel: &NdArray ) -> NdArray {
    // NOTE: Just trying to get an SVD up before implementing other algos
    let mut data = vec![0_f32; 4];
    let a= (kernel.data[1]) / (kernel.data[0] + kernel.data[3]);
    let c = 1_f32 / (1_f32 + a.powi(2)).sqrt();
    let s = (1_f32 - c.powi(2)).sqrt();

    data[0] = c * kernel.data[0];
    data[1] = c * kernel.data[1] - s * kernel.data[3];
    data[2] = s * kernel.data[0];
    data[3] = s * kernel.data[1] + c * kernel.data[3];
    NdArray::new(kernel.dims.clone(), data)    
    // let dims = kernel.dims.clone();
    // data[0] = c;
    // data[1] = -s;
    // data[2] = s;
    // data[3] = c;
    // let double_check = NdArray {dims, data};
    // let check_orth = blas::tensor_mult(2, &double_check, &transpose(double_check.clone()));
    // println!("Check ortho in make symmetric {:?}", check_orth);
    // blas::tensor_mult(2, &double_check, &kernel)
}



fn jacobi_iteration(upper:bool, mut givens:GivensDecomposition) -> GivensDecomposition {
    let rows = givens.kernel.dims[0];
    let cols = givens.kernel.dims[1];
    let rotation:NdArray;
    rotation = jacobi_rotation(0, 1, &givens.kernel);
    let rotation_star = transpose(rotation.clone());
    givens.kernel = blas::tensor_mult(2, &rotation, &givens.kernel);
    givens.kernel = blas::tensor_mult(2, &givens.kernel, &rotation_star);
    // givens.rotation = blas::tensor_mult(2, &rotation, &givens.rotation);

    givens
}

fn jacobi_decomp(mut kernel:NdArray) -> GivensDecomposition {
    let mut upper = true;
    let rotation = blas::create_identity_matrix(kernel.dims[0]);
    let mut givens = GivensDecomposition { rotation, kernel };
    let mut iteration = 4;
    while iteration > 0 {
        iteration -=1;
        upper = !upper;
        givens = jacobi_iteration(upper, givens);
        println!("Jacobi iteration {:?}", givens.kernel);
    }
    givens
}

fn eigen_square_v1(mut a:NdArray) -> NdArray {
    // NOTE: Currently just implementing for 2x2
    assert_eq!(a.data.len(), 4);
    let gamma = a.data[0].powi(2) + a.data[1].powi(2) - a.data[2].powi(2) - a.data[3].powi(2);
    let delta = a.data[1] * a.data[3] + a.data[0] * a.data[2];


    let theta = (-2_f32 * delta).atan2(gamma) / 2.0;
    let c = (theta).cos();
    let s = (theta).sin();
    let tji = s * a.data[0] + c * a.data[2];
    let tjj = s * a.data[1] + c * a.data[3];
    let theta_hat = (tji).atan2(tjj);
    let c_hat = theta_hat.cos();
    let s_hat = theta_hat.sin();

    println!("c {}, s {}, theta {}", c, s, theta);

    // TODO: make this multiplication implicit after it works
    let mut u_rotation= NdArray::new(vec![2;2],vec![0_f32; 4]);
    let mut vt_rotation= NdArray::new(vec![2;2], vec![0_f32; 4]);
    
    u_rotation.data[0]=c;
    u_rotation.data[1]=-s;
    u_rotation.data[2]=s;
    u_rotation.data[3]=c;

    vt_rotation.data[0]=c_hat;
    vt_rotation.data[1]=s_hat;
    vt_rotation.data[2]=-s_hat;
    vt_rotation.data[3]=c_hat;
    
    println!("U rotation {:?}", u_rotation);
    println!("V' rotation {:?}", vt_rotation);

    println!("Check U ortho {:?}",  blas::tensor_mult(1, &u_rotation, &transpose(u_rotation.clone())));
    println!("Check V ortho {:?}",  blas::tensor_mult(1, &vt_rotation, &transpose(vt_rotation.clone())));
    
    a = blas::tensor_mult(2, &u_rotation, &a);
    a = blas::tensor_mult(2, &a, &vt_rotation);

    a
}


fn eigen_square(mut a:NdArray) -> NdArray {
    // NOTE: Currently just implementing for 2x2
    assert_eq!(a.data.len(), 4);
    let gamma = a.data[0].powi(2) + a.data[1].powi(2) - a.data[2].powi(2) - a.data[3].powi(2);
    let delta = a.data[1] * a.data[3] + a.data[0] * a.data[2];


    let theta = (-2_f32 * delta).atan2(gamma) / 2.0;
    let c = (theta).cos();
    let s = (theta).sin();
    let tji = s * a.data[0] + c * a.data[2];
    let tjj = s * a.data[1] + c * a.data[3];
    let theta_hat = (tji).atan2(tjj);
    let c_hat = theta_hat.cos();
    let s_hat = theta_hat.sin();

    println!("c {}, s {}, theta {}", c, s, theta);

    // TODO: make this multiplication implicit after it works
    let mut u_rotation= NdArray::new(vec![2;2],vec![0_f32; 4]);
    let mut vt_rotation= NdArray::new(vec![2;2], vec![0_f32; 4]);
    
    u_rotation.data[0]=c;
    u_rotation.data[1]=-s;
    u_rotation.data[2]=s;
    u_rotation.data[3]=c;

    vt_rotation.data[0]=c_hat;
    vt_rotation.data[1]=s_hat;
    vt_rotation.data[2]=-s_hat;
    vt_rotation.data[3]=c_hat;
    
    println!("U rotation {:?}", u_rotation);
    println!("V' rotation {:?}", vt_rotation);

    println!("Check U ortho {:?}",  blas::tensor_mult(1, &u_rotation, &transpose(u_rotation.clone())));
    println!("Check V ortho {:?}",  blas::tensor_mult(1, &vt_rotation, &transpose(vt_rotation.clone())));
    
    a = blas::tensor_mult(2, &u_rotation, &a);
    a = blas::tensor_mult(2, &a, &vt_rotation);

    a
}





// fn main() {
//     let mut data = vec![0_f32; 4];
//     let mut dims = vec![2; 2];
//     // data[0] = 4_f32;
//     // data[1] = 1_f32;
//     // data[2] = 2_f32;
//     // data[3] = 3_f32;
//     data[0] = 1_f32;
//     data[1] = 2_f32;
//     data[2] = 3_f32;
//     data[3] = 4_f32;
//     let x = blas::NdArray::new(dims, data.clone());
//     println!("x: {:?}", x);
//     let eigen = eigen_square(x);
//     println!("eigen values {:?}", eigen);
// }

fn main() {
    let mut data = vec![0_f32; 4];
    let mut dims = vec![2; 2];
    // data[0] = 1_f32;
    // data[1] = 1_f32;
    // data[2] = 0_f32;
    // data[3] = 1_f32;
    // data[0] = 0_f32;
    // data[1] = -6_f32;
    // data[2] = 1_f32;
    // data[3] = 5_f32;
    data[0] = 2_f32;
    data[1] = -1_f32;
    data[2] = -1_f32;
    data[3] = 3_f32;
    let x = blas::NdArray::new(dims, data.clone());
    println!("x: {:?}", x);
    // let real_schur = real_schur_decomp(x);
    // println!("real schur kernel {:?}", real_schur.kernel);
    // println!("real schur rotation {:?}", real_schur.rotation);
    //
    let y = qr_decompose(x.clone());
    println!("triangle {:?}", y.triangle);


    // let q = real_schur.rotation;
    // let q_star = transpose(q.clone());
    // println!("Schur rotation {:?}", q);
    // let q_orthogonality_check = blas::tensor_mult(2, &q, &q_star);
    // println!("U orthogonality check {:?}", q_orthogonality_check);

    // let symmetric = make_symmetric(&real_schur.kernel);
    // println!("Symmetric values {:?}", symmetric);
    
    // // let eigen = givens_decomp(symmetric);
    // let eigen = eigen_square(y.triangle);
    // let eigen = jacobi_decomp(symmetric);
    // let eigen = givens_decomp(y.triangle);
    let eigen = givens_decomp(x.clone());
    // println!("eigen values {:?}", eigen);
    
    // let eigen = givens_decomp(real_schur.kernel);
    // let eigen = jacobi_decomp(symmetric);
    // println!("eigen values {:?}", eigen.kernel);

}
