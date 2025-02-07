use rand::Rng;

#[derive(Clone, Copy)]
pub enum ActivationFunction {
    Sigmoid,
    Identity,
}

pub struct  Neuron {
    bias: f32,
    weights: Vec<f32>,
    // needed for training
    memory: f32,
    activation: ActivationFunction
}


pub struct Layer {
    neurons: Vec<Neuron>
}


pub struct NeuralNet {
    layers: Vec<Layer>
}


pub fn random_32() -> f32 {
    rand::thread_rng().gen_range(-0.5..0.5)
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
    prediction.iter().zip(result.iter())
        .map(|(p, r)| (p - r) * (p - r))
        .sum()
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

    pub fn fit(&mut self, error:&[f32]) -> Vec<f32> {
        let rate = 0.15_f32;
        let derivative =  self.memory * (1_f32 - self.memory);
        let delta = scalar_product(derivative, &error);
        let update = scalar_product(rate, &delta);

        self.weights = vector_diff(&self.weights, &update);
        self.bias -=  rate * delta.iter().sum::<f32>();
        update
    }
}

impl Layer {
    pub fn new(k:u8, n:u8, activation:ActivationFunction) -> Layer {
        let mut neurons = Vec::with_capacity(k as usize);
        for _ in 0..n {
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
        let mut propegated_errors = vec![];
        for (neuron, error_vec) in self.neurons.iter_mut().zip(errors) {
            let propegated = neuron.fit(error_vec);
            propegated_errors.push(propegated);
        }
        matrix_transpose(&propegated_errors)
    }
}

impl NeuralNet{
    pub fn new(input:u8, dim:Vec<u8>) -> NeuralNet{
        let length  = dim.len();
        let mut layers:Vec<Layer> = Vec::with_capacity(length);
        let touch = Layer::new( input, dim[0], ActivationFunction::Sigmoid );
        layers.push(touch);
        for i in 1..dim.len() {
            let current = Layer::new(dim[i - 1], dim[i], ActivationFunction::Sigmoid);
            layers.push(current);
        }
        // let stream = Layer::new( dim[length - 1], input, ActivationFunction::Identity);
        let stream = Layer::new( dim[length - 1], 1, ActivationFunction::Identity);
        layers.push(stream);
        NeuralNet{ layers }
    }

    pub fn predict(&mut self, mut input:Vec<f32>) -> Vec<f32> {
        for layer in &mut self.layers {
            input = layer.forward(&input);
        }
        input
    }

    pub fn train(&mut self, target:Vec<f32>, actual:Vec<f32>) -> Vec<Vec<f32>> {
        let mut error = vec![vector_diff(&target,&actual)];
        error = matrix_transpose(&error);
        for layer in self.layers.iter_mut().rev() {
            error = layer.backward(&error)
        }
        return error
    }

}

