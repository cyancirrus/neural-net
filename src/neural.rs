#![allow(warnings)]
use rand::Rng;
use std::cmp::min;

const GRADIENT_CLIP_THRESHOLD: f32 = 3.0;
const LEARNING_RATE: f32 = 0.015;
// const LEARNING_RATE: f32 = 0.1;

#[derive(Clone, Copy)]
pub enum ActivationFunction {
    Sigmoid,
    Identity,
}

pub struct  Neuron {
    pub bias: f32,
    pub weights: Vec<f32>,
    // needed for training
    pub memory: f32,
    activation: ActivationFunction
}


pub struct Layer {
    pub neurons: Vec<Neuron>
}


pub struct NeuralNet {
    pub layers: Vec<Layer>
}


pub fn random_32() -> f32 {
    // rand::thread_rng().gen_range(-1.0..1.0)
    let r :f32 = rand::thread_rng().gen_range(-0.75..0.75);
    if r > 0_f32 {
        r + 0.25_f32
    } else {
        r - 0.25_f32
    }
}


pub fn dot_product(x:&[f32], y:&[f32]) -> f32 {
    x.iter().zip(y.iter()).map(|(&x, &y)| x * y).sum()
}

pub fn sigmoid(x:f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn identity(x:f32) -> f32 {
    x
}

pub fn square_matrix_transpose(mut matrix:Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let rows = matrix.len();
    let cols= matrix[0].len();
    for row in (0..rows) {
        for col in (0..cols) {
            if row < cols {
                let aij = matrix[row][col];
                let aji = matrix[col][row];
                matrix[row][col] = aji;
                matrix[col][row] = aij;
            }
        }
    }
    matrix
}

pub fn matrix_transpose(matrix:&Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut transpose = vec![vec![0_f32;rows];cols];
    for row in 0..rows {
        for col in 0..cols {
            transpose[col][row] = matrix[row][col];
        }
    }
    transpose
}

pub fn vector_diff(x:&[f32], y:&[f32]) -> Vec<f32> {
    x.iter().zip(y.iter()).map(|(&x, &y)| x - y).collect()
}

pub fn vector_product(x:&[f32], y:&[f32]) -> Vec<f32> {
    x.iter().zip(y.iter()).map(|(&x, &y)| x * y).collect()
}

pub fn scalar_product(lambda:f32, vector:&[f32]) -> Vec<f32> {
    vector.iter().map(|&vector| lambda * vector).collect()
}
    
pub fn loss_squared(prediction:Vec<f32>, result:Vec<f32>) -> f32 {
    let loss = prediction.iter().zip(result.iter())
        .map(|(p, r)| (p - r) * (p - r))
        .sum();
    loss
}


impl Neuron  {
    pub fn new(n: u8, activation:ActivationFunction) -> Neuron  {
        let bias:f32 = random_32();
        let weights: Vec<f32> = (0..n).map(|_| random_32()).collect();
        let memory: f32 = 0_f32;
        Neuron{ bias, weights, memory, activation }
    }

    pub fn calculate(&mut self, input:&[f32]) -> f32 {
        let product:f32 = dot_product(&self.weights, &input);
        self.memory = match self.activation {
            ActivationFunction::Sigmoid => sigmoid(product+self.bias),
            ActivationFunction::Identity => product + self.bias,
        };
        self.memory
    }
    pub fn derivative(&self) -> f32 {
        match self.activation {
            ActivationFunction::Sigmoid => &self.memory * (1_f32 - &self.memory),
            ActivationFunction::Identity => self.memory.sqrt(),
        }
    }
    pub fn fit(&mut self, error:&[f32]) -> Vec<f32> {
        println!("START");
        println!("ERROR {:?}", error);

        let all_error = error.iter().sum::<f32>();

        let derivative =  self.derivative();
        println!("DERIVATIVE {:?}", derivative);
        let raw_delta = all_error * derivative;
        let delta = GRADIENT_CLIP_THRESHOLD.min(raw_delta);
        let update = scalar_product(LEARNING_RATE * delta, &self.weights);
        // let backwards_error = scalar_product(LEARNING_RATE * raw_delta, &self.weights);
        let backwards_error = scalar_product(delta, &self.weights);

        println!("PREUPDATED weights: {:?}", self.weights);
        self.weights = vector_diff(&self.weights, &update);
        println!("Updated Weights: {:?}", self.weights);
        println!("Updated to bias: {:?}", LEARNING_RATE * all_error);
        self.bias -=  LEARNING_RATE * all_error;
        println!("DONE");
        // println!("----------");
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
        let mut propagated_errors = vec![];
        // println!("This is the total error thingy! {:?}", errors);
        for (neuron, error_vec) in self.neurons.iter_mut().zip(errors) {
            // println!("This is the error vector {:?}", error_vec);
            let propagated = neuron.fit(error_vec);
            propagated_errors.push(propagated);
        }
        matrix_transpose(&propagated_errors)
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
            let stream = Layer::new(1, dim[length - 2], ActivationFunction::Identity);
            layers.push(stream);
        } else {
            let current = Layer::new( dim[0], input, ActivationFunction::Sigmoid);
            let output = Layer::new( 1, dim[0], ActivationFunction::Identity);
            layers.push(current);
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
        let mut error = vec![vector_diff(&actual, &target)];
        // let mut error = vec![vector_diff(&target, &actual)];
        error = matrix_transpose(&error);
        for layer in self.layers.iter_mut().rev() {
            // println!("Neuron length : {}", layer.neurons.len());
            error = layer.backward(&error)
        }
        return error
    }

}

