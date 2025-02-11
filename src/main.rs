#![allow(warnings)]
mod training;
mod neural;
use training::{generate_training_data, load_training_data};
use neural::{ NeuralNet, Neuron, loss_squared };
use std::io::{Write, BufWriter};


fn main() {
    // // Generate training data
    // generate_training_data("training_data/adder.csv", 200);

    // Load training data
    let training_data = load_training_data("training_data/adder.csv");

    let mut nn = NeuralNet::new(2, vec![2, 2, 1]);
    // let mut nn = NeuralNet::new(2, vec![1]);

    // Train the neural network
    let epochs = 10_000;
    // let epochs = 100;
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        for (x, y, z) in &training_data {
            let input = vec![*x, *y];
            let target = vec![*z];
            let prediction = nn.predict(input.clone());

            let loss = loss_squared(prediction.clone(), target.clone());
            // println!("----------------------------------------------");
            // println!("target {:?}, prediction {:?}, loss {:?}", target, prediction, loss);
            total_loss += loss;

            nn.train(target, prediction);
        }
        // println!("Epoch: {}, Loss: {}", epoch, total_loss);

        if epoch % 1000 == 0 {
            println!("Epoch: {}, Loss: {}", epoch, total_loss);
        }
    }

    // Test the neural network on some new data
    // let test_cases = vec![
    //     (1.0, 3.0),
    //     (2.0, 3.0),
    //     (5.0, 7.0),
    //     (1.0, 9.0),
    //     (6.0, 4.0),
    //     (8.0, 5.0),
    // ];
    let test_cases = vec![
        (1.0, 4.0),
        (-8.0, -4.0),
        (3.0, 1.0),
    ];

    println!("\nTesting the trained network on new data:");
    for (x, y) in test_cases {
        let input = vec![x, y];
        let prediction = nn.predict(input);
        // println!("{} + {} = {}", x, y, prediction[0]);
        println!("{} + {} = {}", x, y, prediction[0]);
    }

    println!("----------------------------------------------");
    println!("Hey here are the weights");
    let inspect = nn.layers.last();
    match inspect {
        Some(layer) => {
            for neuron in &layer.neurons {
                println!("Neurons: {:?}", neuron.weights);
                println!("Memory: {:?}", neuron.mem_output);
                println!("Bias: {:?}", neuron.mem_output);
            }
        }
        None => {}
    }
    // for layer in nn.layers {
    //     for neuron in &layer.neurons {
    //         println!("Neurons: {:?}", neuron.weights);
    //         println!("Memory: {:?}", neuron.mem_output);
    //         println!("Bias: {:?}", neuron.mem_output);
    //     }
    // }
    
}
