use std::ops::Mul;

use d2l::utils::*;
use tch::{Tensor, Kind, Device, vision};

const IMAGE_DIM: i64 = 784;
const HIDDEN_NODES: i64 = 128;
const LABELS: i64 = 10;

fn main() {
    tch::manual_seed(42);
    
    let device = Device::cuda_if_available();

    let m = vision::mnist::load_dir("data/mnist").unwrap().to_device(device);
    let lr = 0.1;
    let num_epochs = 1000;
    let batch_size = 500;

    let mut model = MLP::new(IMAGE_DIM, HIDDEN_NODES, LABELS);
    let train_iter = data_iter(batch_size, &m.train_images, &m.train_labels);
    let test_iter = data_iter(0, &m.test_images, &m.test_labels);
    train_ch3_ani(&mut model, &train_iter, &test_iter,  cross_entropy_softmax, lr, num_epochs);
}


#[derive(Debug)]
struct MLP {
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
}

impl MLP {
    fn new(in_dim: i64, hidden_dim: i64, out_dim: i64) -> MLP {
        let w1 = Tensor::randn([in_dim, hidden_dim], (Kind::Float, Device::cuda_if_available())).mul(0.01).set_requires_grad(true);
        let b1 = Tensor::zeros([hidden_dim], (Kind::Float, Device::cuda_if_available())).set_requires_grad(true);
        let w2 = Tensor::randn([hidden_dim, out_dim], (Kind::Float, Device::cuda_if_available())).mul(0.01).set_requires_grad(true);
        let b2 = Tensor::zeros([out_dim], (Kind::Float, Device::cuda_if_available())).set_requires_grad(true);
        MLP{w1, b1, w2, b2}
    }
}

impl Model for MLP {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        let h = x.matmul(&self.w1) + &self.b1;
        let h = relu(&h);
        h.matmul(&self.w2) + &self.b2
    }

    fn update(&mut self, batch_size: i64, lr: f64) {
        let batch_size = batch_size as usize;
        sgd(vec![&mut self.w2, &mut self.b2, &mut self.w1, &mut self.b1], lr, batch_size)
    }
}

fn relu(x: &Tensor) -> Tensor {
    let a = x.zeros_like();
    x.max_other(&a)
}