extern crate rand;
use rand::Rng;

pub struct  Neuron {
    bias: f32,
    weights: Vec<f32>
}


pub struct Layer {
    neurons: Vec<Neuron>
}


pub struct NeuralNet {
    layers: Vec<Layer>
}


pub fn random_32() -> f32 {
    rand::thread_rng().gen_range(-1.0..1.0)
}


pub fn dot_product(x:&[f32], y:&[f32]) -> f32 {
    x.iter().zip(y.iter()).map(|(&x, &y)| x * y).sum()
}

pub fn sigmoid(x:f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
    // 1.0  / (1.0 + std::f32::consts::E.powf(-unscaled))
} 

impl Neuron  {
    fn new(n:u8) -> Neuron  {
        let bias:f32 = random_32();
        let weights: Vec<f32> = (0..n).map(|_| random_32()).collect();
        Neuron{ bias, weights }
    }

    fn calculate(&self, input:&[f32]) -> f32 {
        let product:f32 = dot_product(&self.weights, &input);
        let a = sigmoid(product + self.bias);
        println!("intermediate {:?}", a);
        sigmoid(product + self.bias)

    }
}

impl Layer {
    pub fn new(k:u8, n:u8) -> Layer {
        let mut neurons = Vec::with_capacity(k as usize);
        for _ in 0..n {
            neurons.push(Neuron::new(k));
        }
        Layer { neurons }
    }
    pub fn forward(&self, input:&[f32]) -> Vec<f32> {
        self.neurons. iter().
            map(|neuron| neuron.calculate(input)).
            collect()
    }
}

impl NeuralNet{
    pub fn new(input:u8, dim:Vec<u8>) -> NeuralNet{
        let length  = dim.len();
        let mut layers:Vec<Layer> = Vec::with_capacity(length);
        let touch = Layer::new( input, dim[0] );
        layers.push(touch);
        for i in 1..dim.len() {
            let current = Layer::new(dim[i - 1], dim[i] );
            layers.push(current);
        }
        let stream = Layer::new( dim[length - 1], input);
        layers.push(stream);
        NeuralNet{ layers }
    }

    pub fn predict(&self, mut input:Vec<f32>) -> Vec<f32> {
        for layer in &self.layers {
            input = layer.forward(&input);
        }
        input
    }

    // pub fn classify(&self, input:Vec<f32>) -> f32 {
    //     let ones = vec![1.0; input.len()];
    //     let prelim_out = self.predict(input);
    //     let product = dot_product(&ones, &prelim_out);
    //     sigmoid(product)
    // }
}

fn main() {
    let input:Vec<f32> = (0..5).map(|_| random_32()).collect();
    
    // Test dot product
    let n:Neuron = Neuron::new(5);
    let test = n.calculate(&input);
    println!("Test should be between [0, 1] {}", test);

    // Test prediction
    let dims = vec![1,2,3,4,5, 2];
    let nn = NeuralNet::new(5, dims);
    let test_predict = nn.predict(input);
    println!("Predict should be [0, 1] {:?}", test_predict);
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
