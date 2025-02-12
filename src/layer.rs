use crate::math;
use crate::neural;
use rayon::prelude::{ParallelIterator};
use rayon::prelude::*;

trait LayerTrait {
    fn forward(&mut self, input:&[f32]) -> &[f32];
    fn backward(&mut self, error:Vec<Vec<f32>>) -> Vec<Vec<f32>>;
}

struct DenseLayer {
    neurons: Vec<neural::Neuron>,
    mem_input: Vec<f32>,
    mem_output: Vec<f32>,
}
struct SoftmaxLayer {
    neurons: Vec<neural::Neuron>,
    mem_input: Vec<f32>,
    mem_output: Vec<f32>,
}
struct SoftmaxIndexLayer {
    neurons: Vec<neural::Neuron>,
    mem_input: Vec<f32>,
    mem_output: Vec<f32>,
}
struct CausalLayer {
    neurons: Vec<neural::Neuron>,
    mem_input: Vec<f32>,
    mem_output: Vec<f32>,
}
    
pub fn calculate(input:&[f32], neuron:&neural::Neuron) -> f32 {
        let product:f32 = math::dot_product(&neuron.weights, &input);
        match neuron.activation {
            neural::ActivationFunction::Sigmoid => math::sigmoid( product+neuron.bias ),
            neural::ActivationFunction::Identity => product + neuron.bias,
            neural::ActivationFunction::Softmax => ( product + neuron.bias ).exp()
        }
    }


impl LayerTrait for DenseLayer {
    fn forward(&mut self, input:&[f32]) -> &[f32] {
        self.mem_input = input.to_vec();
        self.mem_output = self.neurons.par_iter()
            .map(|neuron| calculate(input, neuron)).
            collect();
        &self.mem_output

    }
    fn backward(&mut self, errors:Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let mut propogated_error = self.neurons
            .par_iter_mut()
            .zip(errors.par_iter())
            .map(|(neuron, error)| neuron.fit(&error))
            .collect();

        math::matrix_transpose(&propogated_error)
    }
}

fn cross_apply(x:&[f32], y:&[f32], f_enum:fn(usize, f32, usize, f32) -> f32) -> Vec<Vec<f32>> {
    let rows = x.len();
    let cols = y.len();
    let mut matrix = vec![vec![0_f32;cols];rows];

    for row in 0..rows {
        for col in 0..cols {
            matrix[row][col] = f_enum(row, x[row], col, y[col]);
        }
    }
    matrix
}

fn cross_product(x:Vec<f32>, y:Vec<f32>) -> Vec<Vec<f32>> {
    fn product(_:usize, x:f32, _:usize, y:f32) -> f32 {
        x * y
    }
    cross_apply(&x, &y, product)
}
    
fn derivative_structure(x_index:usize, x:f32, y_index:usize, y:f32) -> f32 {
    let kronecker_delta = 1_f32;
    x* ( kronecker_delta - y)
}

impl LayerTrait for SoftmaxLayer {
    fn forward(&mut self, input:&[f32]) -> &[f32] {
        let prelim = self.neurons
            .par_iter()
            .map(| neuron | calculate(input, neuron));

        let denominator:&f32 = &prelim.clone().sum::<f32>();
        self.mem_input = input.to_vec();
        self.mem_output = prelim.map(|prelim| prelim / denominator).collect();
        &self.mem_output
    }
    fn backward(&mut self, error:Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let jacobian = cross_apply(&self.mem_output, &self.mem_output, derivative_structure);
        math::matrix_transpose(&jacobian)
        
    }
}
