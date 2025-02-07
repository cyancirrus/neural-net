extern crate rand;
use rand::Rng;
use std::fs::File;
use std::io::{Write, BufWriter, BufRead, BufReader};


pub fn generate_training_data(filename: &str, num_samples: usize) {
    let mut rng = rand::thread_rng();
    let file = File::create(filename).expect("Failed to create file");
    let mut writer = BufWriter::new(file);

    for _ in 0..num_samples {
        let x: f32 = rng.gen_range(0.0..5.0);
        let y: f32 = rng.gen_range(0.0..5.0);
        // let x: f32 = rng.gen_range(0.0..10.0);
        // let y: f32 = rng.gen_range(0.0..10.0);
        // let z = x + y;  // True output
        let z = x - y;  // True output
        writeln!(writer, "{},{},{}", x, y, z).expect("Failed to write to file");
    }
    println!("Training data saved to {}", filename);
}

pub fn load_training_data(filename: &str) -> Vec<(f32, f32, f32)> {
    let file = File::open(filename).expect("Failed to open file");
    let reader = BufReader::new(file);
    let mut data = Vec::new();

    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        let parts: Vec<f32> = line.split(',')
            .map(|s| s.trim().parse().expect("Invalid number"))
            .collect();
        if parts.len() == 3 {
            data.push((parts[0], parts[1], parts[2]));
        }
    }
    data
}
