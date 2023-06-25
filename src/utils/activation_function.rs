use tch::{Tensor, Kind};

pub fn softmax(x: &Tensor) -> Tensor {
    let x_exp = x.exp();
    let partition = x_exp.sum_dim_intlist(1, true, Kind::Float);
    x_exp / partition
}