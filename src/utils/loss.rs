use tch::{Tensor, Kind};

use super::softmax;

pub fn squared_loss(y_hat: &Tensor, y: &Tensor) -> Tensor {
    let size = y_hat.size();
    (y_hat - y.reshape(size)).pow(&Tensor::from(2)) / 2
}

pub fn cross_entropy(y_hat: &Tensor, y: &Tensor) -> Tensor {
    let x = Tensor::arange(y_hat.size()[0], (Kind::Int64, y.device()));
    -y_hat.index(&[Some(&x), Some(y)]).log()
}

pub fn cross_entropy_softmax(y_hat: &Tensor, y: &Tensor) -> Tensor {
    let x = Tensor::arange(y_hat.size()[0], (Kind::Int64, y.device()));
    -softmax(y_hat).index(&[Some(&x), Some(y)]).log()
}