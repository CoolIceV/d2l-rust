use std::{ops::Mul, iter::Zip};

use d2l::utils::*;
use tch::{ Kind, Tensor, Device, IndexOp};

fn main() {
    let x = Tensor::from_slice2(&[[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]]).to_device(Device::cuda_if_available());
    pool2d(&x, &[2, 2], "max").print();
    pool2d(&x, &[2, 2], "avg").print();

}

fn pool2d(x: &Tensor, pool_size: &[i64; 2], mode: &str) -> Tensor {
    let p_h = pool_size[0];
    let p_w = pool_size[1];
    let y = Tensor::zeros(&[x.size()[0] - p_h + 1, x.size()[1] - p_w + 1], (Kind::Float, Device::cuda_if_available()));
    for i in 0..y.size()[0] {
        for j in 0..y.size()[1] {
            let mut v = x.slice(0, i, i + p_h, 1).slice(1, j, j + p_w, 1);
            if mode == "max" {
                v = v.max();
            } else if mode == "avg" {
                v = v.mean(Kind::Float);
            }
            y.i((i, j)).copy_(&v);
        }
    }
    y
}