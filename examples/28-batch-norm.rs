use std::ops::Mul;

use d2l::utils::Model;
use tch::{Tensor, Kind, Device};

fn main() {

}

struct BarchNorm {
    gamma: Tensor,
    beta: Tensor,
    moving_mean: Tensor,
    moving_var: Tensor,
    eps: f64,           //1e-5
    momentum: f64,      //0.9
    training: bool,
}

impl BarchNorm {
    fn new(num_features: i64, num_dims: i64, eps: f64, momentum: f64) -> Self {
        let gamma = if num_dims == 2 {
            Tensor::ones([1, num_features], (Kind::Float, Device::cuda_if_available()))
        } else {
            Tensor::ones([1, num_features, 1, 1], (Kind::Float, Device::cuda_if_available()))
        };

        let beta = gamma.zeros_like();
        let moving_mean = gamma.zeros_like();
        let moving_var = gamma.ones_like();

        BarchNorm { gamma, beta, moving_mean, moving_var, eps, momentum, training: false }
    }
}

impl Model for BarchNorm {
    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn forward(&mut self, x: &Tensor) -> Tensor {
        let (y, moving_mean, moving_var) = batch_norm(x, &self.gamma, &self.beta, &self.moving_mean, &self.moving_var, self.eps, self.momentum, self.training);
        self.moving_mean = moving_mean.to_device(Device::cuda_if_available());
        self.moving_var = moving_var.to_device(Device::cuda_if_available());
        y
    }

    fn update(&mut self, batch_size: i64, lr: f64) {}
}


fn batch_norm(x: &Tensor, gamma: &Tensor, beta: &Tensor, moving_mean: &Tensor, moving_var: &Tensor, eps: f64, momentum: f64, is_training: bool) -> (Tensor, Tensor, Tensor) {
    let mut x_hat = Tensor::zeros_like(x);
    let mut moving_mean = moving_mean.copy();
    let mut moving_var = moving_var.copy();

    if !is_training {
         x_hat = (x - moving_mean.copy()) / (moving_var.copy() + eps);
    } else {
        let mean: Tensor;
        let var: Tensor;
        if x.size().len() == 2 {
            mean = x.mean_dim(0, true, Kind::Float);
            var = ((x - &mean).pow(&Tensor::from(2))).mean_dim(0, true, Kind::Float);
        } else {
            mean = x.mean_dim(&vec![0, 2, 3], true, Kind::Float);
            var = ((x - &mean).pow(&Tensor::from(2))).mean_dim(&vec![0, 2, 3], true, Kind::Float);
        }

        x_hat = (x - &mean) / (&var + eps).sqrt();

        moving_mean = moving_mean * momentum + (mean * (1.0 - momentum));
        moving_var = moving_var * momentum + (var * (1.0 - momentum));
    }

    (gamma * x_hat + beta, moving_mean, moving_var)

}