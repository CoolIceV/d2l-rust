use std::ops::Mul;

use d2l::utils::*;
use tch::{ Kind, Tensor, Device, nn::Sgd, vision};

fn dropout_layer(x: &Tensor, dropout: f64) -> Tensor {
    if dropout == 1. {
        Tensor::zeros(x.size(), (Kind::Float, Device::cuda_if_available()))
    } else  if dropout == 0. {
        x.shallow_clone()
    } else {
        let mask = Tensor::ones(x.size(), (Kind::Float, Device::cuda_if_available())).uniform(0., 1.).greater(dropout);
        x * mask / (1. - dropout)
        // mask
    }
}

#[derive(Debug)]
struct MLP {
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
    w3: Tensor,
    b3: Tensor,
    dropout1: f64,
    dropout2: f64,
    training: bool,
}

impl MLP {
    fn new(in_dim: i64, hidden_dim1: i64, hidden_dim2: i64, out_dim: i64, dropout1: f64, dropout2: f64) -> MLP {
        let w1 = Tensor::randn([in_dim, hidden_dim1], (Kind::Float, Device::cuda_if_available())).mul(0.01).set_requires_grad(true);
        let b1 = Tensor::zeros([hidden_dim1], (Kind::Float, Device::cuda_if_available())).set_requires_grad(true);
        let w2 = Tensor::randn([hidden_dim1, hidden_dim2], (Kind::Float, Device::cuda_if_available())).mul(0.01).set_requires_grad(true);
        let b2 = Tensor::zeros([hidden_dim2], (Kind::Float, Device::cuda_if_available())).set_requires_grad(true);
        let w3 = Tensor::randn([hidden_dim2, out_dim], (Kind::Float, Device::cuda_if_available())).mul(0.01).set_requires_grad(true);
        let b3 = Tensor::zeros([out_dim], (Kind::Float, Device::cuda_if_available())).set_requires_grad(true);
        MLP{w1, b1, w2, b2, w3, b3, dropout1, dropout2, training: true, }
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

impl Model for MLP {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        let h = x.matmul(&self.w1) + &self.b1;
        let mut h = relu(&h);
        if self.training {
            h = dropout_layer(&h, self.dropout1);
        }
        let h = h.matmul(&self.w2) + &self.b2;
        let mut h = relu(&h);
        if self.training {
            h = dropout_layer(&h, self.dropout2);
        }
        h.matmul(&self.w3) + &self.b3
    }

    fn update(&mut self, batch_size: i64, lr: f64) {
        let batch_size = batch_size as usize;
        sgd(vec![&mut self.w2, &mut self.b2, &mut self.w1, &mut self.b1, &mut self.w3, &mut self.b3], lr, batch_size)
    }
}

fn main() {
    // let x = Tensor::arange(16, (Kind::Float, Device::cuda_if_available())).reshape([2, 8]);
    // x.print();
    // dropout_layer(&x, 0.).print();
    // dropout_layer(&x, 0.5).print();
    // dropout_layer(&x, 1.).print();

    tch::manual_seed(42);
    
    let device = Device::cuda_if_available();

    let m = vision::mnist::load_dir("data/mnist").unwrap().to_device(device);
    let lr = 0.5;
    let num_epochs = 10;
    let batch_size = 256;
    let dropout1 = 0.2;
    let dropout2 = 0.1;
    let mut model = MLP::new(784, 256, 256, 10, dropout1, dropout2);
    let train_iter = data_iter(batch_size, &m.train_images, &m.train_labels);
    let test_iter = data_iter(0, &m.test_images, &m.test_labels);
    train_ch3_ani(&mut model, &train_iter, &test_iter,  cross_entropy_softmax, lr, num_epochs);
}