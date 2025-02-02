extern crate rand;

use rand::Rng;

pub struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}
pub struct Layer {
    neurons: Vec<Neuron>,
}
pub struct Network {
    layers: Vec<Layer>,
}
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl Neuron {
    pub fn new(size: usize) -> Self {
        let mut rng = rand::thread_rng();
        Neuron {
            weights: (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect(),
            bias: rng.gen_range(-1.0..1.0),
        }
    }
    pub fn forward(&self, inputs: &Vec<f64>) -> f64 {
        let dotproduct: f64 = self
            .weights
            .iter()
            .zip(inputs.iter())
            .map(|(w, v)| w * v)
            .sum();
        sigmoid(dotproduct + self.bias)
    }
}

impl Layer {
    pub fn new(num_neurons: usize, num_inputs_per_neuron: usize) -> Self {
        let neurons = (0..num_neurons)
            .map(|_| Neuron::new(num_inputs_per_neuron))
            .collect();
        Layer { neurons }
    }
    pub fn forward(&self, inputs: &Vec<f64>) -> Vec<f64> {
        self.neurons.iter().map(|n| n.forward(inputs)).collect()
    }
}
    
fn loss_squared(prediction:Vec<f64>, result:Vec<f64>) -> f64 {
    prediction.iter().zip(result.iter())
        .map(|(p, r)| (p - r) * (p - r))
        .sum()
}

impl Network {
    pub fn new(size: Vec<usize>) -> Self {
        let mut layers = Vec::with_capacity(size.len());
        for i in 1..size.len() {
            let num_inputs = size[i - 1];
            let num_neurons = size[i];
            layers.push(Layer::new(num_neurons, num_inputs));
        }
        Network { layers }
    }

    pub fn predict(&self, input: Vec<f64>) -> Vec<f64> {
        let mut output = input;
        for layer in &self.layers {
            output = layer.forward(&output)
        }
        output
    }

}
