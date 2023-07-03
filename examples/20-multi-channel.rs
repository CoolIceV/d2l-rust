use std::{ops::Mul, iter::Zip};

use d2l::utils::*;
use tch::{ Kind, Tensor, Device, IndexOp};

fn main() {
    let x = Tensor::from_slice(&[0., 1., 2., 3., 4., 5., 6., 7., 8.,
                1., 2., 3., 4., 5., 6., 7., 8., 9.]).reshape([2, 3, 3]);
    let k = Tensor::from_slice(&[0., 1., 2., 3., 1., 2., 3., 4.]).reshape([2, 2, 2]);

    x.print();
    k.print();
    corr2d_multi_in(&x, &k).print();

    let k = Tensor::stack(&[k.copy(), k.copy()+1, k.copy()+2], 0);
    corr2d_multi_in_out(&x, &k).print();


    let x = Tensor::zeros(&[3, 3, 3], (Kind::Float, Device::cuda_if_available())).normal_(0., 1.);
    let k = Tensor::zeros(&[2, 3, 1, 1], (Kind::Float, Device::cuda_if_available())).normal_(0., 1.);
    
    corr2d_multi_in_out_1x1(&x, &k).print();
    corr2d_multi_in_out(&x, &k).print();
}


fn corr2d(x: &Tensor, k: &Tensor) -> Tensor {
    let h = k.size()[0] as i64;
    let w = k.size()[1] as i64;
    let y = Tensor::zeros(&[x.size()[0] - h + 1, x.size()[1] - w + 1], (Kind::Float, Device::cuda_if_available()));
    for i in 0..y.size()[0] {
        for j in 0..y.size()[1] {
            y.i((i, j)).copy_(&x.slice(0, i, i + h, 1).slice(1, j, j + w, 1).mul(k).sum(Kind::Float));
        }
    }
    y
}

fn corr2d_multi_in(x: &Tensor, k: &Tensor) -> Tensor {
    let mut y = Tensor::zeros(&[x.size()[1] - k.size()[1] + 1, x.size()[2] - k.size()[2] + 1], (Kind::Float, Device::cuda_if_available()));
    for i in 0..x.size()[0] {
        y += corr2d(&x.i(i), &k.i(i));
    }
    y
}

fn corr2d_multi_in_out(x: &Tensor, k: &Tensor) -> Tensor {
    let mut y = Vec::new();
    // Tensor::stack(tensors, 0)
    for i in 0..k.size()[0] {
        y.push(corr2d_multi_in(&x, &k.i(i)));
    }
    Tensor::stack(&y, 0)
}

fn corr2d_multi_in_out_1x1(x: &Tensor, k: &Tensor) -> Tensor {
    let c_i = x.size()[0];
    let h = x.size()[1];
    let w = x.size()[2];
    
    let c_o = k.size()[0];
    
    let x = x.reshape(&[c_i,h * w]);
    let k = k.reshape(&[c_o, c_i]);
    let y = k.matmul(&x);
    y.reshape(&[c_o, h, w])
}