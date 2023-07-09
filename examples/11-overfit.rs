use std::ops::Mul;

use d2l::utils::*;
use tch::{vision, Kind, Tensor, Device, IndexOp};
use special::Gamma;

#[derive(Debug)]
struct Linear{
    ws: Tensor,
    // bs: Tensor,
}

impl Linear {
    fn new(in_dim: i64, out_dim: i64) -> Linear {
        let ws = Tensor::zeros([in_dim, out_dim], (Kind::Float, Device::cuda_if_available())).set_requires_grad(true);
        // let bs = Tensor::zeros([out_dim], (Kind::Float, Device::cuda_if_available())).set_requires_grad(true);
        // Linear{ws, bs}
        Linear{ws}
    }

}

impl Model for Linear {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        x.matmul(&self.ws)
        // x.matmul(&self.ws) + &self.bs
    }

    fn update(&mut self, batch_size: i64, lr: f64) {
        let batch_size = batch_size as usize;
        // sgd(vec![&mut self.ws, &mut self.bs], lr, batch_size);
        sgd(vec![&mut self.ws], lr, batch_size);
    }
}

fn main() {

    tch::manual_seed(43);

    let max_degree = 20;
    let n_train = 100;
    let n_test = 300;

    let mut true_w = Tensor::zeros(&[max_degree], (Kind::Float, Device::cuda_if_available()));
    true_w.slice(0, 0, 4, 1).copy_(&Tensor::from_slice(&[5., 1.2, -3.4, 5.6]));

    let features = Tensor::zeros(&[n_train + n_test, 1], (Kind::Float, Device::cuda_if_available())).normal_(0., 2.);

    let mut poly_features = features.pow(&Tensor::arange(max_degree, (Kind::Float, Device::cuda_if_available())).reshape(&[1, -1]));
    
    
    // poly_features.print();

    for i in 0..max_degree {
        poly_features.slice(1, i, i+1, 1).copy_(&(poly_features.slice(1, i, i+1, 1) / ((i+1) as f64).gamma()));
    }

    let mut labels = poly_features.matmul(&true_w);



    labels += Tensor::zeros(labels.size(), (Kind::Float, Device::cuda_if_available())).normal_(0., 0.9);


    let train_features = poly_features.i((0..n_train, ..2));

    let train_labels = labels.i(0..n_train);
    let test_features = poly_features.i((n_train.., ..2));
    let test_labels = labels.i(n_train..);

    let mut model = Linear::new(test_features.size()[1] ,1);

    let batch_size = 100;

    let train_iter = data_iter(batch_size, &train_features, &train_labels);
    let test_iter = data_iter(batch_size, &test_features, &test_labels);
    let mut a = Animator::new(&["train loss", "test loss"]);
    for epoch in 0..1000 {
        let (train_l, train_acc) = train_epoch_ch3(&mut model, &train_iter, squared_loss, 0.01);
        let train_loss = evaluate_loss(&mut model, &train_iter, squared_loss);
        let test_loss = evaluate_loss(&mut model, &test_iter, squared_loss);
        println!("epoach: {}, train loss:{}, test_loss:{}", epoch, train_loss, test_loss);
        a.add_point("train loss", epoch as f64, train_loss);
        a.add_point("test loss", epoch as f64, test_loss);
        a.draw();
    }
    true_w.print();
    model.ws.print();
    while true {}
}
