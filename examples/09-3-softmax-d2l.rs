use tch::{vision, Kind, Tensor, Device};
use d2l::utils::*;

const IMAGE_DIM: i64 = 784;
const LABELS: i64 = 10;

#[derive(Debug)]
struct Linear{
    ws: Tensor,
    bs: Tensor,
}

impl Linear {
    fn new(in_dim: i64, out_dim: i64) -> Linear {
        let ws = Tensor::zeros([in_dim, out_dim], (Kind::Float, Device::cuda_if_available())).set_requires_grad(true);
        let bs = Tensor::zeros([out_dim], (Kind::Float, Device::cuda_if_available())).set_requires_grad(true);
        Linear{ws, bs}
    }

}

impl Model for Linear {
    fn forward(&self, x: &Tensor) -> Tensor {
        softmax(&(x.matmul(&self.ws) + &self.bs))
    }

    fn update(&mut self, batch_size: i64, lr: f64) {
        let batch_size = batch_size as usize;
        sgd(vec![&mut self.ws, &mut self.bs], lr, batch_size);
    }
}

fn main() {
    tch::manual_seed(42);
    
    let device = Device::cuda_if_available();
    let m = vision::mnist::load_dir("data/mnist").unwrap().to_device(device);

    let mut model = Linear::new(IMAGE_DIM, LABELS);

    let lr = 0.3;
    let num_epochs = 1000;
    let batch_size = 500;

    let train_iter = data_iter(batch_size, &m.train_images, &m.train_labels);
    let test_iter = data_iter(0, &m.test_images, &m.test_labels);
    train_ch3(&mut model, &train_iter, &test_iter,  cross_entropy, lr, num_epochs);
}