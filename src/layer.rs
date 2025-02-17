#![allow(warnings)]
use crate::math;
use crate::neural;
use itertools::multizip;
use rayon::prelude::ParallelIterator;
use rayon::prelude::*;

#[derive(Clone, Copy)]
pub enum ActivationFunction {
    Sigmoid,
    Identity,
    ReLu,
}

pub enum MetaLayer {
    Dense(DenseLayer),
    Softmax(SoftmaxLayer),
    Causal(CausalLayer),
}

trait LayerTrait {
    fn new(&self, n: usize, k: usize, activate: ActivationFunction) -> MetaLayer;
    fn forward(&mut self, input: &[f32]) -> &[f32];
    fn backward(&mut self, error: Vec<f32>) -> Vec<f32>;
}

impl LayerTrait for MetaLayer {
    fn new(&self, n: usize, k: usize, activate: ActivationFunction) -> MetaLayer {
        match self {
            MetaLayer::Dense(DenseLayer) => DenseLayer::new(&DenseLayer, n, k, activate),
            MetaLayer::Softmax(SoftmaxLayer) => SoftmaxLayer::new(&SoftmaxLayer, n, k, activate),
            MetaLayer::Causal(CausalLayer) => CausalLayer::new(&CausalLayer, n, k, activate),
        }
    }
    fn forward(&mut self, input: &[f32]) -> &[f32] {
        match self {
            MetaLayer::Dense(layer) => layer.forward(input),
            MetaLayer::Softmax(layer) => layer.forward(input),
            MetaLayer::Causal(layer) => layer.forward(input),
        }
    }
    fn backward(&mut self, error: Vec<f32>) -> Vec<f32> {
        match self {
            MetaLayer::Dense(layer) => layer.backward(error),
            MetaLayer::Softmax(layer) => layer.backward(error),
            MetaLayer::Causal(layer) => layer.backward(error),
        }
    }
}

pub fn derivative(predictions: &[f32], activation: ActivationFunction) -> Vec<f32> {
    match activation {
        ActivationFunction::Sigmoid => predictions
            .par_iter()
            .map(|predict| predict * (1_f32 - predict))
            .collect(),
        ActivationFunction::Identity => {
            vec![1_f32; predictions.len()]
        }
        ActivationFunction::ReLu => predictions
            .par_iter()
            .map(|predict| if *predict > 0_f32 { 1_f32 } else { 0_f32 })
            .collect(),
    }
}

struct DenseLayer {
    neurons: Vec<neural::Neuron>,
    activate: ActivationFunction,
    mem_input: Vec<f32>,
    mem_output: Vec<f32>,
    mem_derivative: Vec<f32>,
}
struct SoftmaxLayer {
    neurons: Vec<neural::Neuron>,
    mem_input: Vec<f32>,
    mem_output: Vec<f32>,
    mem_derivative: Vec<f32>,
    activate: ActivationFunction,
}
struct CausalLayer {
    neurons: Vec<neural::Neuron>,
    activate: ActivationFunction,
    mem_derivative: Vec<f32>,
    mem_input: Vec<f32>,
    mem_output: Vec<f32>,
}

impl LayerTrait for DenseLayer {
    fn new(&self, n: usize, k: usize, activate: ActivationFunction) -> MetaLayer {
        let mut neurons = Vec::with_capacity(k as usize);
        for i in 0..k {
            neurons.push(neural::Neuron::new(n, activate));
        }
        let mem_input = vec![0_f32; n];
        let mem_output = vec![0_f32; n];
        let mem_derivative = vec![0_f32; n];
        MetaLayer::Dense(DenseLayer {
            neurons,
            activate,
            mem_input,
            mem_output,
            mem_derivative,
        })
    }
    fn forward(&mut self, input: &[f32]) -> &[f32] {
        self.mem_input = input.to_vec();
        self.mem_output = self
            .neurons
            .par_iter()
            .map(|neuron| neuron.calculate(input))
            .collect();
        &self.mem_output
    }
    fn backward(&mut self, errors: Vec<f32>) -> Vec<f32> {
        let mut propogated_error: Vec<Vec<f32>> = self
            .neurons
            .par_iter_mut()
            .zip(self.mem_derivative.par_iter())
            .zip(errors.par_iter())
            .map(|((neuron, error), derivative)| neuron.fit(error, derivative, &self.mem_input))
            .collect();

        let transpose = math::matrix_transpose(propogated_error);
        collapse(transpose)
    }
}

fn derivative_structure(i: usize, j: usize, x: f32, y: f32) -> f32 {
    if i == j {
        x * (1_f32 - y)
    } else {
        -x * y
    }
}

fn jacobian(x: &[f32]) -> Vec<Vec<f32>> {
    let length = x.len();
    let mut matrix = vec![vec![0_f32; length]; length];
    for i in 0..length {
        for j in 0..length {
            matrix[j][i] = derivative_structure(i, j, x[i], x[j])
        }
    }
    matrix
}

fn collapse(matrix: Vec<Vec<f32>>) -> Vec<f32> {
    return matrix.par_iter().map(|x| x.into_iter().sum()).collect();
}

impl SoftmaxLayer {
    fn derivative(&mut self, predict: &[f32]) -> Vec<f32> {
        let transpose = jacobian(predict);
        collapse(transpose)
    }
}

impl LayerTrait for SoftmaxLayer {
    fn new(&self, n: usize, k: usize, activate: ActivationFunction) -> MetaLayer {
        let neurons: Vec<neural::Neuron> = Vec::with_capacity(0 as usize);
        let mem_input = vec![0_f32; n];
        let mem_output = vec![0_f32; n];
        let mem_derivative = vec![0_f32; n];
        MetaLayer::Softmax(SoftmaxLayer {
            neurons,
            activate,
            mem_input,
            mem_output,
            mem_derivative,
        })
    }
    fn forward(&mut self, input: &[f32]) -> &[f32] {
        let prelim = input.par_iter();
        let denominator: &f32 = &prelim.clone().sum::<f32>();
        self.mem_input = input.to_vec();
        self.mem_output = prelim.map(|prelim| prelim / denominator).collect();
        &self.mem_output
    }
    fn backward(&mut self, error: Vec<f32>) -> Vec<f32> {
        let transpose = jacobian(&error);
        collapse(transpose)
    }
}

impl LayerTrait for CausalLayer {
    fn new(&self, n: usize, k: usize, activate: ActivationFunction) -> MetaLayer {
        let mut neurons: Vec<neural::Neuron> = Vec::with_capacity(k as usize);
        for i in 0..k {
            neural::Neuron::new(i, activate);
        }
        let mem_input = vec![0_f32; n];
        let mem_output = vec![0_f32; n];
        let mem_derivative = vec![0_f32; n];
        MetaLayer::Causal(CausalLayer {
            neurons,
            activate,
            mem_input,
            mem_output,
            mem_derivative,
        })
    }
    fn forward(&mut self, input: &[f32]) -> &[f32] {
        self.mem_input = input.to_vec();
        self.mem_output = self
            .neurons
            .par_iter()
            .map(|neuron| neuron.calculate(&input[0..neuron.i]))
            .collect();
        &self.mem_output
    }
    fn backward(&mut self, errors: Vec<f32>) -> Vec<f32> {
        let mut propogated_error: Vec<Vec<f32>> = self
            .neurons
            .par_iter_mut()
            .zip(self.mem_derivative.par_iter())
            .zip(errors.par_iter())
            .map(|((neuron, error), derivative)| {
                neuron.fit(error, derivative, &self.mem_input[0..neuron.i])
            })
            .collect();

        let transpose = math::matrix_transpose(propogated_error);
        collapse(transpose)
    }
}
