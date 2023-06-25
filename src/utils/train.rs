#![cfg_attr(debug_assertions, allow(dead_code))]

use super::model::Model;
use tch::{Tensor, Kind};
use tch::IndexOp;

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

fn evaluate_accuracy<T>(model: &T, data_iter: &Vec<(Tensor, Tensor)>) -> f64 
where T: Model {
    tch::no_grad(|| {
        let mut metric = Accumulator::new(2);

        for (x, y) in data_iter {
            let y_hat = model.forward(&x);
            metric.add(vec![accuracy(&y_hat, &y), y.size()[0] as f64]);
        }

        metric.get(0) / metric.get(1)
    })
}

pub fn train_epoch_ch3<T>(model: &mut T, train_iter: &Vec<(Tensor, Tensor)>, loss: fn (y_hat: &Tensor, y: &Tensor) -> Tensor, lr: f64) -> (f64, f64)
where T: Model {
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