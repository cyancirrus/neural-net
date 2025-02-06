extern crate rand;
mod neural;
use neural::{ NeuralNet, Neuron, random_32 };


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
