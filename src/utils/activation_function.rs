#![cfg_attr(debug_assertions, allow(dead_code, unused_must_use))]

use tch::{Tensor, Kind};

pub fn softmax(x: &Tensor) -> Tensor {
    let x_exp = x.exp();
    let partition = x_exp.sum_dim_intlist(1, true, Kind::Float);
    x_exp / partition
}

fn relu(x: &Tensor) -> Tensor {
    let a = x.zeros_like();
    x.max_other(&a)
}