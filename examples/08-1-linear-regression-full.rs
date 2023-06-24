use std::ops::Add;

use tch::{Tensor, Kind, Device};
use d2l::utils::dataset::data_iter;

fn synthetic_data(w: Vec<f32>, b: f32, num: i32) -> (Tensor, Tensor){
    let mut x = Tensor::ones([num as i64, w.len() as i64], (Kind::Float, Device::Cpu));
    let x = x.normal_(0.0, 1.0);

    let mut w = Tensor::from_slice(&w);
    let y = x.matmul(&w.t_()).add(Tensor::from(b));
    let mut noize = Tensor::ones(y.size(), (Kind::Float, Device::Cpu));

    let y = y+ noize.normal_(0.1, 0.1); 
    (x, y)
}

fn main() {
    let (features, labels) = synthetic_data(vec![2.0, -3.4], 4.2, 10);

    let mut w = Tensor::ones([2, 1], (Kind::Float, Device::Cpu));
    let w = w.normal_(0.0, 0.01);
    let mut w = w.set_requires_grad(true);
    let b = Tensor::ones(1, (Kind::Float, Device::Cpu));
    let mut b = b.set_requires_grad(true);
    let lr = 0.3;
    let num_epochs = 3;
    let batch_size = 3;
    let net = linreg;
    let loss = squared_loss;


    for epoch in 0..num_epochs {
        for (x, y) in data_iter(batch_size, &features, &labels) {
            let l = loss(net(&x, &w, &b), &y);
            l.sum(Kind::Float).backward();
            sgd(vec![&mut w, &mut b], lr, batch_size);
        }

       print_epoch(loss, net, epoch, &features, &w, &b, &labels)
       
    }
}

fn print_epoch(loss: fn( Tensor, &Tensor) -> Tensor, net: fn(&Tensor,  &Tensor, &Tensor) -> Tensor , epoch: usize, features: &Tensor, w: &Tensor, b: &Tensor, labels: &Tensor) {
    tch::no_grad(||{
        let train_l = loss(net(&features, &w, &b), &labels);
        println!("========================================");
        println!("epoch {}, loss {}", epoch, train_l.mean(Kind::Float));
        println!("w: {}", w);
        println!("b: {}", b);
    })
}

fn linreg(x: &Tensor, w: &Tensor, b: &Tensor) -> Tensor {
    x.matmul(&w) + b
}

fn squared_loss(y_hat: Tensor, y: &Tensor) -> Tensor {
    let size = y_hat.size();
    (y_hat - y.reshape(size)).pow(&Tensor::from(2)) / 2
}

fn sgd(params: Vec<&mut Tensor>, lr: f64, batch_size: usize) {
    tch::no_grad(|| {
        for param in params.into_iter() {
            *param -= lr * param.grad() / (batch_size as f64);
            _ = param.grad().zero_();
        } 
    })
}