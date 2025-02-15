
fn tensor_mult(blocksize:usize, x:NdArray, y:NdArray) -> NdArray {
    assert!(blocksize > 0);
    // don't know how to do this without cloning seems easy compare?
    // assert_eq!(&x.dims, &y.dims.rev().collect::<Vec<usize>>(), "dimension mismatch");
    let blocksize:usize = 2;
    // fixed matrix size ~ M[4, 4] for prototyping
    let mut new:Vec<f32> = vec![0_f32; 4 * 4];
    let mut value: f32;
    let x_rows = 4;
    let x_cols = 4;
    let y_rows = 4;
    let y_cols = 4;

    // iterate by blocksize
    for i in (0..4).step_by(blocksize) {
        for j in (0..4).step_by(blocksize) {
            for k in 0..blocksize {
                for ii in 0..blocksize {
                    for jj in 0..blocksize {
                        for kk in 0..blocksize + x_cols % blocksize {
                            // println!("kk: {}", kk);
                            // println!("left input index : {}", (i + ii ) * x_rows  + k);
                            // println!("right input index : {}", (j + jj ) * y_cols + k);
                            // println!("output i: {}", i);
                            // println!("output ii: {}", ii);
                            // println!("accum: {}", accum);
                            // println!("k: {}", k);
                            println!("blocksize: {}", blocksize);
                            // let index = (i + ii) * x_rows +  k * blocksize + kk;
                            let index = (i + ii + blocksize) * x_rows +  k * blocksize + kk;
                            let index = (i + ii + blocksize) * x_rows +  k * blocksize + kk;
                            // new[index] += accum;
                            let value ={
                                x.data[(i + ii ) * x_rows + k * blocksize + kk]
                                * y.data[(j + jj) * y_cols + k * blocksize + kk]
                            };
                            println!("i: {}, ii: {}, k:{}, kk:{}, index: {}, value: {}", i, ii, k, kk, index, value);
                            new[index] += value;
                        }
                        // let index = (i) * x_rows + j * y_cols + (jj  + ii)* blocksize;
                        // let index = (i + ii) * x_rows + k * blocksize + kk;
                        // println!("output index : {}", index);
                        // println!("output i: {}", i);
                        // println!("output ii: {}", ii);
                        // println!("output j: {}", j);
                        // println!("output jj: {}", jj);
                        // println!("Value: {}", accum);
                        // new[index] += accum;
                        // println!("Value: {}", accum);
                        // accum = 0_f32;
                        // accum = 0_f32;
                    }
                }
            }
        }
    }
    // Need to do like x.dim[1] = y.dim[x] and clone from x
    NdArray::new ( x.dims, new )
}
