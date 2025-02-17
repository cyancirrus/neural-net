#![allow(warnings)]
use crate::math;

use rand::Rng;
use std::cmp::min;
use rayon::prelude::{ParallelIterator};
use rayon::prelude::*;


const GRADIENT_CLIP_THRESHOLD: f32 = 4.0;
const BIAS_CLIP_THRESHOLD: f32 = 4.0;
const LEARNING_RATE: f32 = 0.0020;

#[derive(Clone, Copy)]
pub enum ActivationFunction {
    Sigmoid,
    Identity,
}

pub struct  Neuron {
    pub bias: f32,
    pub weights: Vec<f32>,
    // needed for training
    pub mem_output: f32,
    pub mem_input: Vec<f32>,
    activation: ActivationFunction
}


pub struct Layer {
    pub neurons: Vec<Neuron>
}


pub struct NeuralNet {
    pub layers: Vec<Layer>
}

pub fn random_32() -> f32 {
    rand::thread_rng().gen_range(-1.0..1.0)
    // let r :f32 = rand::thread_rng().gen_range(-0.75..0.75);
    // if r > 0_f32 {
    //     r + 0.25_f32
    // } else {
    //     r - 0.25_f32
    // }
}

fn bias_clip(x:f32) -> f32 {
    x.clamp(-BIAS_CLIP_THRESHOLD, BIAS_CLIP_THRESHOLD)
}

fn gradient_clip(x:f32) -> f32 {
    x.clamp(-GRADIENT_CLIP_THRESHOLD, GRADIENT_CLIP_THRESHOLD)
}

impl Neuron  {
    pub fn new(n: u8, activation:ActivationFunction) -> Neuron  {
        let bias:f32 = random_32();
        let weights: Vec<f32> = (0..n).map(|_| random_32()).collect();
        let mem_output: f32 = 0_f32;
        let mem_input: Vec<f32> = vec![0_f32; n as usize];
        Neuron{ bias, weights, mem_output, mem_input, activation }
    }

    pub fn calculate(&mut self, input:&[f32]) -> f32 {
        let product:f32 = math::dot_product(&self.weights, &input);
        self.mem_output = match self.activation {
            ActivationFunction::Sigmoid => math::sigmoid(product+self.bias),
            ActivationFunction::Identity => product + self.bias,
        };
        self.mem_input = input.to_vec();
        self.mem_output
    }
    pub fn derivative(&self) -> f32 {
        match self.activation {
            ActivationFunction::Sigmoid => &self.mem_output * (1_f32 - &self.mem_output),
            ActivationFunction::Identity => 1_f32,
        }
    }
    pub fn fit(&mut self, error:&[f32]) -> Vec<f32> {
        let all_error = error.iter().sum::<f32>();
        let derivative =  self.derivative();
        let raw_delta = all_error * derivative;
        let delta = gradient_clip(raw_delta);
        let update = math::scalar_product(LEARNING_RATE * delta, &self.mem_input);
        let backwards_error = math::scalar_product( delta, &self.weights);

        self.weights = math::vector_diff(&self.weights, &update);
        self.bias -=  LEARNING_RATE * bias_clip(all_error) / 3.0_f32;
        backwards_error
    }
}

impl Layer {
    pub fn new(k:u8, n:u8, activation:ActivationFunction) -> Layer {
        let mut neurons = Vec::with_capacity(k as usize);
        for _ in 0..k {
            neurons.push(Neuron::new(n, activation));
        }
        Layer { neurons }
    }
    pub fn forward(&mut self, input:&[f32]) -> Vec<f32> {
        self.neurons.iter_mut().
            map(|neuron| neuron.calculate(input)).
            collect()
    }
    pub fn backward(&mut self, errors:&[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut propogated_errors = self.neurons
            .par_iter_mut() 
            .zip(errors.par_iter())
            .map(|(neuron, error_vec)| neuron.fit(error_vec))
            .collect()
        ;
        math::matrix_transpose(&propogated_errors)
    }
}

impl NeuralNet{
    pub fn new(input:u8, dim:Vec<u8>) -> NeuralNet{
        let length  = dim.len();
        let mut layers:Vec<Layer> = Vec::with_capacity(length);
        if length > 1 {
            let touch = Layer::new( dim[0], input, ActivationFunction::Sigmoid );
            layers.push(touch);
            for i in 1..dim.len() - 1 {
                let current = Layer::new(dim[i], dim[i - 1], ActivationFunction::Sigmoid);
                layers.push(current);
            }
            let stream = Layer::new(dim[length -1], dim[length - 2], ActivationFunction::Identity);
            layers.push(stream);
        } else {
            let output = Layer::new(dim[0], input, ActivationFunction::Identity);
            layers.push(output);
        }
        NeuralNet{ layers }
    }

    pub fn predict(&mut self, mut input:Vec<f32>) -> Vec<f32> {
        for layer in &mut self.layers {
            input = layer.forward(&input);
        }
        input
    }

    pub fn train(&mut self, target:Vec<f32>, actual:Vec<f32>) -> Vec<Vec<f32>> {
        let mut error = vec![math::vector_diff(&actual, &target)];
        error = math::matrix_transpose(&error);
        for layer in self.layers.iter_mut().rev() {
            error = layer.backward(&error)
        }
        return error
    }

}

