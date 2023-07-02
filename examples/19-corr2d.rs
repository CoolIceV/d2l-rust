use std::ops::Mul;

use d2l::utils::*;
use tch::{ Kind, Tensor, Device, IndexOp};

struct Conv2D {
    w: Tensor,
    // b: Tensor,
}

impl Conv2D {
    fn new(kernel_size: &[i64]) -> Self {
        Conv2D { 
            w: Tensor::rand(kernel_size, (Kind::Float, Device::cuda_if_available())).set_requires_grad(true), 
            // b: Tensor::zeros(&[1], (Kind::Float, Device::cuda_if_available())), 
        }
    }
}

impl Model for Conv2D {
    fn forward(&self, x: &Tensor) -> Tensor {
        // corr2d(x, &self.w) + &self.b
        corr2d(x, &self.w)
    }

    fn update(&mut self, batch_size: i64, lr: f64) {
        // sgd(vec![&mut self.w, &mut self.b], lr, batch_size as usize)
        sgd(vec![&mut self.w], lr, batch_size as usize)
    }
}

fn main() {
    let x = Tensor::from_slice2(&[[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]]);
    let k = Tensor::from_slice2(&[[0., 1.], [2., 3.]]);
    corr2d(&x, &k).print();

    let x = Tensor::ones(&[6, 8], (Kind::Float, Device::cuda_if_available()));
    x.slice(1, 2, 6, 1).copy_(&x.slice(1, 2, 6, 1).zeros_like());
    x.print();

    let k = Tensor::from_slice2(&[[1., -1.]]).to_device(Device::cuda_if_available());
    let y = corr2d(&x, &k);
    y.print();

    let mut conv2d = Conv2D::new(&[1, 2]);
    let x = &x.reshape(&[6, 8]);
    let y = &y.reshape(&[6, 7]);

    let lr = 0.03;
    for _ in 0..50 {
        // let y_hat = model.forward(&x);
        let y_hat = conv2d.forward(&x);
        let l = (y - y_hat).pow(&Tensor::from(2)).sum(Kind::Float) / 2.;
        l.sum(Kind::Float).backward();
        conv2d.update(1, lr);
    }

    conv2d.w.print();
    // conv2d.b.print();
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