#![cfg_attr(debug_assertions, allow(dead_code))]

use super::Animator;
use super::model::Model;
use tch::{Tensor, Kind, nn};
use tch::IndexOp;
use tch::nn::{Module, OptimizerConfig};

fn accuracy(y_hat: &Tensor, y: &Tensor) -> f64 {
    let y_hat = if y_hat.size().len() > 1 && y_hat.size()[1] > 1 {
        y_hat.argmax(1, false)
    } else {
        y_hat.i(..)
    };

    let cmp = y_hat.eq_tensor(y);
    cmp.to_kind(y.kind()).sum(Kind::Float).double_value(&[])
}

#[derive(Debug)]
struct Accumulator {
    data: Vec<f64>,
}

impl Accumulator {
    fn new(x: usize) -> Self {
        Accumulator {
            data: vec![0.0; x],
        }
    }

    fn reset(&mut self) {
        self.data = vec![0.0; self.data.len()];
    }

    fn add(&mut self, args: Vec<f64>) {
        for (i, arg) in args.iter().enumerate() {
            self.data[i] += arg;
        }
    }

    fn get(&self, i: usize) -> f64 {
        self.data[i]
    }
}

fn evaluate_accuracy<T>(model: &mut T, data_iter: &Vec<(Tensor, Tensor)>) -> f64 
where T: Model {
    model.set_training(false);
    tch::no_grad(|| {
        let mut metric = Accumulator::new(2);

        for (x, y) in data_iter {
            let y_hat = model.forward(&x);
            metric.add(vec![accuracy(&y_hat, &y), y.size()[0] as f64]);
        }

        metric.get(0) / metric.get(1)
    })
}


pub fn evaluate_loss<T>(model: &mut T, data_iter: &Vec<(Tensor, Tensor)>, loss: fn (y_hat: &Tensor, y: &Tensor) -> Tensor) -> f64 
where T: Model {
    model.set_training(false);
    tch::no_grad(|| {
        let mut metric = Accumulator::new(2);

        for (x, y) in data_iter {
            let y_hat = model.forward(&x);
            let y = y.reshape_as(&y_hat);
            let l = loss(&y_hat, &y);
            metric.add(vec![l.sum(Kind::Float).double_value(&[]), l.numel() as f64]);
        }

        metric.get(0) / metric.get(1)
    })
}

pub fn train_epoch_ch3<T>(model: &mut T, train_iter: &Vec<(Tensor, Tensor)>, loss: fn (y_hat: &Tensor, y: &Tensor) -> Tensor, lr: f64) -> (f64, f64)
where T: Model {
    model.set_training(true);
    let mut metric = Accumulator::new(3);
    for (x, y) in train_iter {
        let y_hat = model.forward(&x);
        let l = loss(&y_hat, &y);
        l.sum(Kind::Float).backward();
        model.update(x.size()[0], lr);
        metric.add(vec![l.sum(Kind::Float).double_value(&[]), accuracy(&y_hat, &y), y.size()[0] as f64]);
    }

    //loss, train acc
    (metric.get(0) / metric.get(2), metric.get(1) / metric.get(2))
}

pub fn train_ch3<T>(model: &mut T, train_iter: &Vec<(Tensor, Tensor)>, test_iter: &Vec<(Tensor, Tensor)>, loss: fn (y_hat: &Tensor, y: &Tensor) -> Tensor, lr: f64, num_epochs: usize) 
where T: Model {
    for epoch in 0..num_epochs {
        let (train_l, train_acc) = train_epoch_ch3(model, train_iter, loss, lr);
        let test_acc = evaluate_accuracy(model, test_iter);
        println!("epoch {:4}, train loss {:8.5}, train acc {:5.2}%, test acc {:5.2}%", epoch, train_l, 100. * train_acc, 100. * test_acc);
    }
}

pub fn train_ch3_ani<T>(model: &mut T, train_iter: &Vec<(Tensor, Tensor)>, test_iter: &Vec<(Tensor, Tensor)>, loss: fn (y_hat: &Tensor, y: &Tensor) -> Tensor, lr: f64, num_epochs: usize) 
where T: Model {
    let mut a = Animator::new(&["loss", "train acc", "test acc"]);
    for epoch in 0..num_epochs {
        let (train_l, train_acc) = train_epoch_ch3(model, train_iter, loss, lr);
        let test_acc = evaluate_accuracy(model, test_iter);
        a.add_point("loss", epoch as f64, train_l);
        a.add_point("train acc", epoch as f64, train_acc);
        a.add_point("test acc", epoch as f64, test_acc);
        a.draw();
        println!("epoch {:4}, train loss {:8.5}, train acc {:5.2}%, test acc {:5.2}%", epoch, train_l, 100. * train_acc, 100. * test_acc);
    }
}


pub fn train_epoch_ch6(net: &nn::Sequential, train_iter: &Vec<(Tensor, Tensor)>, opt:&mut nn::Optimizer) -> (f64, f64) {
    // net.set_training(true);
    let mut metric = Accumulator::new(3);
    for (x, y) in train_iter {
        let loss = net.forward(&x).cross_entropy_for_logits(&y);
        
        opt.backward_step(&loss);

        tch::no_grad(|| {
            let loss = net.forward(&x).cross_entropy_for_logits(&y);
            let train_accuracy = net.forward(&x).accuracy_for_logits(&y);
            metric.add(vec![f64::try_from(&loss).unwrap(), f64::try_from(&train_accuracy).unwrap(), 1.]);
        });

    }

    //loss, train acc
    (metric.get(0) / metric.get(2), metric.get(1) / metric.get(2))
}


pub fn train_ch6(net: &nn::Sequential, train_iter: &Vec<(Tensor, Tensor)>, test_iter: &Vec<(Tensor, Tensor)>, num_epochs: usize, opt:&mut nn::Optimizer) {
    for epoch in 0..num_epochs {
        let (train_l, train_acc) = train_epoch_ch6(net, train_iter, opt);
        let test_acc = evaluate_accuracy_ch6(net, test_iter);
        println!("epoch {:4}, train loss {:8.5}, train acc {:5.2}%, test acc {:5.2}%", epoch, train_l, 100. * train_acc, 100. * test_acc);
    }
}

fn evaluate_accuracy_ch6(model: &nn::Sequential, data_iter: &Vec<(Tensor, Tensor)>) -> f64 {
    tch::no_grad(|| {
        let mut metric = Accumulator::new(2);

        for (x, y) in data_iter {
            let y_hat = model.forward(&x);
            metric.add(vec![accuracy(&y_hat, &y), y.size()[0] as f64]);
        }

        metric.get(0) / metric.get(1)
    })
}