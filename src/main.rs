mod list;
mod tree;
pub use list::List;
pub use tree::Tree;
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
    pub fn new(size: usize) -> Self {
        Layer {
            neurons: (0..size).map(|_| Neuron::new(size)).collect(),
        }
    }
}

// impl Network {
//     fn new(size:Vec<usize>) -> Self {

//     }
// }

fn main() {
    let mut tree = Tree::new();
    tree.append(10);
    tree.append(5);
    tree.append(15);
    tree.append(1);
    tree.append(100);
    tree.display();

    let mut a = List::new();
    a.append(32);
    a.append(100);
    a.display();
}
