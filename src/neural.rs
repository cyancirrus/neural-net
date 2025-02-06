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
    
fn loss_squared(prediction:Vec<f64>, result:Vec<f64>) -> f64 {
    prediction.iter().zip(result.iter())
        .map(|(p, r)| (p - r) * (p - r))
        .sum()
}


impl Neuron  {
    pub fn new(n:u8) -> Neuron  {
        let bias:f32 = random_32();
        let weights: Vec<f32> = (0..n).map(|_| random_32()).collect();
        Neuron{ bias, weights }
    }

    pub fn calculate(&self, input:&[f32]) -> f32 {
        let product:f32 = dot_product(&self.weights, &input);
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

