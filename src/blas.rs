use rayon::prelude::*;

pub struct Matrix {
    pub rows:usize,
    pub cols:usize,
    pub data:Vec<f32>,
}

pub struct NdArray {
    pub dims:Vec<usize>,
    pub data:Vec<f32>,
}
    
impl Matrix {
    pub fn new(rows:usize, cols:usize, data:Vec<f32>) -> Matrix {
        assert!(rows >= 0, "rows is not greater than or equal to 0");
        assert!(cols >= 0, "cols is not greater than or equal to 0");
        assert_eq!(data.len(), rows * cols, "dimension mismatch in matrix");
        Matrix { rows, cols, data }
    }
}
impl NdArray {
    pub fn new(dims:Vec<usize>, data:Vec<f32>) -> NdArray {
        NdArray { dims, data }
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
fn column_iterator(rows:usize, cols:usize) -> Vec<usize> {
    let length:usize = rows * cols;
    let mut index:usize;
    (0..length).collect::<Vec<usize>>().par_iter()
        .map(|i| i * cols % length + i / rows)
        .collect::<Vec<usize>>()
}


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

    // iterate by blocksize
    for i in (0..4).step_by(blocksize) {
        for j in (0..y_cols).step_by(blocksize) {
            for k in 0..blocksize {
                for ii in 0..blocksize {
                    for jj in 0..blocksize {
                        for kk in 0..blocksize + x_cols % blocksize {
                            let index = (i + ii) * x_rows +  k * blocksize + kk;
                            let x_index = (i + ii ) * x_rows + j + jj;
                            let y_index =  (j + jj) * y_cols +  k * blocksize + kk;
                            let value ={
                                x.data[x_index]
                                * y.data[y_index]
                            };
                            if index == 4 {
                                println!("x index: {}, x value: {}", x_index, x.data[x_index]);
                                println!("y index: {}, y_value: {}", y_index, y.data[y_index]);
                                println!("index: {}, value: {}, i: {}, ii: {}, j:{}, jj:{}, k:{}, kk:{}, ", index, value, i, ii, j, jj, k, kk, );
                            };
                            new[index] += value;
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
