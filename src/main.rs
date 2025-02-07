mod training;
mod neural;
use training::{generate_training_data, load_training_data};
use neural::{ NeuralNet, Neuron, loss_squared };
use std::io::{Write, BufWriter};
extern crate rand;


// fn main() {
//     generate_training_data("training_data/adder.csv", 500);
// }

fn main() {
    // Generate training data
    generate_training_data("training_data/adder.csv", 500);

    // Load training data
    let training_data = load_training_data("training_data/adder.csv");

    // Create a neural network with [2, 2, 1]
    // let mut nn = NeuralNet::new(2, vec![ 1]);
    let mut nn = NeuralNet::new(2, vec![2, 1]);

    // Train the neural network
    let epochs = 10000;
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        for (x, y, z) in &training_data {
            let input = vec![*x, *y];
            let target = vec![*z];
            let prediction = nn.predict(input.clone());

            let loss = loss_squared(prediction.clone(), target.clone());
            total_loss += loss;

            nn.train(target, prediction);
        }
        println!("Epoch: {}, Loss: {}", epoch, total_loss);

        // if epoch % 1000 == 0 {
        //     println!("Epoch: {}, Loss: {}", epoch, total_loss);
        // }
    }

    // Test the neural network on some new data
    let test_cases = vec![
        (2.0, 3.0),
        (5.0, 7.0),
        (1.0, 9.0),
        (6.0, 4.0),
        (8.0, 5.0),
    ];

    println!("\nTesting the trained network on new data:");
    for (x, y) in test_cases {
        let input = vec![x, y];
        let prediction = nn.predict(input);
        println!("{} + {} = {}", x, y, prediction[0]);
    }
}



// fn main() {
//     let input:Vec<f32> = (0..5).map(|_| random_32()).collect();
    
//     // Test dot product
//     let n:Neuron = Neuron::new(5);
//     let test = n.calculate(&input);
//     println!("Test should be between [0, 1] {}", test);

//     // Test prediction
//     let dims = vec![1,2,3,4,5, 2];
//     let nn = NeuralNet::new(5, dims);
//     let test_predict = nn.predict(input);
//     println!("Predict should be [0, 1] {:?}", test_predict);
// }
