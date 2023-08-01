use std::ops::{Mul, Add};
use tch::{ Kind, Tensor, Device, IndexOp};

fn main() {
    let x = Tensor::from_slice2(&[[0., 1.], [2., 3.]]).to_device(Device::cuda_if_available());
    let k = Tensor::from_slice2(&[[0., 1.], [2., 3.]]).to_device(Device::cuda_if_available());
    trans_conv(&x, &k).print();

    // x.conv_transpose2d(&k, None, 2, 1, 0, 0, 0);
} 

fn trans_conv(x: &Tensor, k: &Tensor) -> Tensor {
    let h = k.size()[0] as i64;
    let w = k.size()[1] as i64;
    let y = Tensor::zeros(&[x.size()[0] + h - 1, x.size()[1] + w - 1], (Kind::Float, Device::cuda_if_available()));
    for i in 0..x.size()[0] {
        for j in 0..x.size()[1] {
            y.i((i..i+h, j..j+w)).copy_(&y.i((i..i+h, j..j+w)).add(&x.i((i, j)).mul(k)));
        }
    }

    y
}