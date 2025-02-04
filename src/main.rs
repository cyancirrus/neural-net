extern crate rand;
use std::ops::Mul;
use std::iter::Sum;
use rand::Rng;

#[derive(Copy, Clone, Debug)]
pub struct  Neuron <const N:usize>{
    bias: f32,
    weights:[f32;N],
}


#[derive(Copy, Clone, Debug)]
pub struct Layer<const N:usize, const K:usize> {
    neurons: [Neuron<N>; K],
}


pub fn random_32() -> f32 {
    rand::thread_rng().gen::<f32>() 
}


pub fn dot_product<const N:usize>(x:&[f32;N], y:&[f32;N]) -> f32 {
    x.iter().zip(y.iter()).map(|(&x, &y)| x * y).sum()
}

pub fn sigmoid(unscaled:f32) -> f32 {
    1.0  / (1.0 + std::f32::consts::E.powf(-unscaled))
} 

impl <const N:usize> Neuron <N> {
    fn new() -> Neuron <N> {
        let bias:f32 = random_32();
        let weights:[f32;N] = std::array::from_fn(|_| random_32());
        Neuron{ bias, weights }
    }

    fn calculate(&self, input:[f32;N]) -> f32 {
        let product = dot_product(&self.weights, &input);
        sigmoid(product + self.bias)

    }
}

impl <const N:usize, const K: usize> Layer <N, K> {
    pub fn new() -> Layer<N, K> {
        let neurons: [Neuron<N>; K] = std::array::from_fn(|_| Neuron::new());
        Layer { neurons }
    }
    pub fn print(&self) {
        println!("{:?}", self.neurons);
    }
}

fn main() {
    let mut n:Neuron<5> = Neuron::new();
    let input:[f32; 5] = std::array::from_fn(|_| random_32());
    let test = n.calculate(input);
    println!("Test should be between [0, 1] {}", test);
    
}


// mod list;
// mod tree;
// mod neural;
// pub use list::List;
// pub use tree::Tree;
// pub use neural::Network;


// fn main() {
//     let input = vec![0.5, 0.1, 0.3]; // Example input
//     let network = Network::new(vec![3, 4, 2]); // 3 inputs, 4 neurons in layer 1, 2 neurons in layer 2
//     let output = network.predict(input);
//     println!("{:?}", output);

//     let mut tree = Tree::new();
//     tree.append(10);
//     tree.append(5);
//     tree.append(15);
//     tree.append(1);
//     tree.append(100);
//     tree.display();

//     let mut a = List::new();
//     a.append(32);
//     a.append(100);
//     a.display();
// }
