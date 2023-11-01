extern crate csv;
use std::error::Error;
use std::path::Path;

fn read_csv<P: AsRef<Path>>(path: P) -> Result<Dataset, Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<f64> = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let feature_vec: Vec<f64> = record
            .iter()
            .take(record.len() - 1)  
            .map(|e| e.parse().unwrap())
            .collect();

        let label: f64 = record[record.len() - 1].parse()?;
        features.push(feature_vec);
        labels.push(label);
    }

    Ok(Dataset { features, labels })
}

pub struct LogisticRegression {
    coefficients: Vec<f64>,
    learning_rate: f64,
}

pub struct Dataset {
    features: Vec<Vec<f64>>,
    labels: Vec<f64>,
}
fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}
impl LogisticRegression {
    pub fn cost_function(&self, dataset: &Dataset) -> f64 {
        let mut total_cost = 0.0;
        let m = dataset.features.len() as f64;  
        let eps = 1e-10;
        for i in 0..dataset.features.len() {
            let mut z = self.coefficients[0];  
            for j in 0..dataset.features[i].len() {
                z += self.coefficients[j + 1] * dataset.features[i][j];
            }
            let mut h = sigmoid(z);
            h = h.max(eps).min(1.0 - eps);

            let y = dataset.labels[i];
            let cost = -y * h.ln() - (1.0 - y) * (1.0 - h).ln();

            total_cost += cost;
        }

        total_cost /= m;

        total_cost
    }
    pub fn gradient_descent(&mut self, dataset: &Dataset) {
        let mut gradients = vec![0.0; self.coefficients.len()];
        let m = dataset.features.len() as f64;  

        for i in 0..dataset.features.len() {
            let mut z = self.coefficients[0];  
            for j in 0..dataset.features[i].len() {
                z += self.coefficients[j + 1] * dataset.features[i][j];
            }

            let h = sigmoid(z);

            let error = h - dataset.labels[i];
            gradients[0] += error;  
            for j in 0..dataset.features[i].len() {
                gradients[j + 1] += error * dataset.features[i][j];
            }
        }

        for j in 0..self.coefficients.len() {
            self.coefficients[j] -= self.learning_rate * gradients[j] / m;
        }
    }
    pub fn predict(&self, features: &Vec<f64>) -> f64 {
        let mut z = self.coefficients[0];  
        for j in 0..features.len() {
            z += self.coefficients[j + 1] * features[j];
        }

        let h = sigmoid(z);

        if h >= 0.5 { 1.0 } else { 0.0 }
    }
    pub fn evaluate(&self, dataset: &Dataset) -> f64 {
        let mut correct_predictions = 0.0;

        for i in 0..dataset.features.len() {
            let predicted_label = self.predict(&dataset.features[i]);
            if predicted_label == dataset.labels[i] {
                correct_predictions += 1.0;
            }
        }

        correct_predictions / (dataset.features.len() as f64)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let dataset = read_csv("diabetes.csv")?;
    
    let input_size = dataset.features[0].len();
    let learning_rate = 0.01;
    let mut model = LogisticRegression {
        coefficients: vec![0.0; input_size + 1],  
        learning_rate,
    };

    let epochs = 1000;
    for epoch in 1..=epochs {
        model.gradient_descent(&dataset);
        
        if epoch % 100 == 0 {
            let cost = model.cost_function(&dataset);
            println!("Epoch {}: Cost = {}", epoch, cost);
        }
    }

    let accuracy = model.evaluate(&dataset);
    println!("Model Accuracy: {}", accuracy);

    let sample_features = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let prediction = model.predict(&sample_features);
    println!("Prediction for sample features: {}", prediction);
    
    Ok(())
}

