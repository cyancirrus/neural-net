use rayon::prelude::*;

pub struct Matrix {
    pub rows:usize,
    pub cols:usize,
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
