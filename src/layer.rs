use crate::math;
use crate::neural;
use itertools::multizip;
use rayon::prelude::{ParallelIterator};
use rayon::prelude::*;

#[derive(Clone, Copy)]
pub enum ActivationFunction {
    Sigmoid,
    Identity,
    ReLu,
}

pub enum Layer {
    DenseLayer,
    Softmax,
}
    
pub fn derivative(predictions:&[f32], activation:ActivationFunction) -> Vec<f32> {
    match activation {
        ActivationFunction::Sigmoid => {
            predictions.par_iter().map(
                |predict| predict * (1_f32 - predict)
                ).collect()
        },
        ActivationFunction::Identity => {
            vec![1_f32;predictions.len()]
        },
        ActivationFunction::ReLu => {
            predictions.par_iter()
                .map(|predict| {
                    if *predict > 0_f32 {
                        1_f32
                    } else {
                        0_f32
                    }
                }) .collect()
        },
    }
}

pub fn calculate(input:&[f32], neuron:&neural::Neuron) -> f32 {
    match neuron.activation {
        ActivationFunction::Sigmoid => {
            let product:f32 = math::dot_product(&neuron.weights, &input);
            math::sigmoid( product+neuron.bias )
        },
        ActivationFunction::Identity => {
            let product:f32 = math::dot_product(&neuron.weights, &input);
            product + neuron.bias
        },
        ActivationFunction::ReLu => {
            let product = math::vector_product(&neuron.weights, &input);
            let filtered = product.par_iter().map(
                |predict| { predict.max(0_f32) }
            ).sum::<f32>();
            filtered + neuron.bias.max(0_f32)
        },
    }
}

trait LayerTrait {
    fn forward(&mut self, input:&[f32]) -> &[f32];
    fn backward(&mut self, error:Vec<f32>) -> Vec<f32>;
}

struct DenseLayer {
    neurons: Vec<neural::Neuron>,
    activation: ActivationFunction,
    mem_input: Vec<f32>,
    mem_output: Vec<f32>,
    mem_derivative: Vec<f32>,
}
struct SoftmaxLayer {
    neurons: Vec<neural::Neuron>,
    mem_input: Vec<f32>,
    mem_output: Vec<f32>,
    mem_derivative: Vec<f32>,
    activation: ActivationFunction,
}
struct SoftmaxIndexLayer {
    neurons: Vec<neural::Neuron>,
    activation: ActivationFunction,
    mem_input: Vec<f32>,
    mem_output: Vec<f32>,
    mem_derivative: Vec<f32>,
}
struct CausalLayer {
    neurons: Vec<neural::Neuron>,
    activation: ActivationFunction,
    mem_derivative: Vec<f32>,
    mem_input: Vec<f32>,
    mem_output: Vec<f32>,
}


impl LayerTrait for DenseLayer {
    fn forward(&mut self, input:&[f32]) -> &[f32] {
        self.mem_input = input.to_vec();
        self.mem_output = self.neurons.par_iter()
            .map(|neuron| calculate(input, neuron)).
            collect();
        &self.mem_output
    }
    fn backward(&mut self, errors:Vec<f32>) -> Vec<f32> {
        let mut propogated_error: Vec<Vec<f32>>  = self.neurons
            .par_iter_mut()
            .zip(self.mem_derivative.par_iter())
            .zip(errors.par_iter())
            .map(|((neuron, error), derivative)| {
                neuron.fit(error, derivative, &self.mem_input)
                })
            .collect();
    
        let transpose = math::matrix_transpose(propogated_error);
        collapse(transpose)
    }
}

    
fn derivative_structure(i:usize, j:usize, x:f32,  y:f32) -> f32 {
    if i == j {
        x * (1_f32 - y)
    } else {
        - x * y
    }
}

fn jacobian(x:&[f32]) -> Vec<Vec<f32>> {
    let length = x.len();
    let mut matrix = vec![vec![0_f32;length];length];
    for i in 0..length {
        for j in 0..length {
            matrix[j][i] = derivative_structure(i, j, x[i], x[j])
        }
    }
    matrix
}

fn collapse(matrix: Vec<Vec<f32>>) -> Vec<f32> {
    return matrix.par_iter()
        .map(|x| x.into_iter().sum())
        .collect()
}

impl SoftmaxLayer {
    fn derivative(&mut self, predict:&[f32]) -> Vec<f32> {
        let transpose = jacobian(predict);
        collapse(transpose)
    }
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
    fn backward(&mut self, error:Vec<f32>) -> Vec<f32> {
        let transpose = jacobian(&error);
        collapse(transpose)
    }
}
