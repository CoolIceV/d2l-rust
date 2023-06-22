use tch::{Tensor, Kind, Device};

fn func(i: &Tensor) -> Tensor {
    let x = i * 2;
    x
}

fn main() {
    let x = Tensor::range(1, 4,  (Kind::Float, Device::cuda_if_available()));
    let x = x.set_requires_grad(true);

    let y:Tensor = 2*x.dot(&x);

    y.backward();

    x.grad().print();

    let x = x.grad().zero();
    let x = x.set_requires_grad(true);
    let y:Tensor = 2*x.dot(&x);
    let u = y.detach();
    let y = x.dot(&x) + u;
    let y = y.set_requires_grad(true);
    let c = func(&y);
    c.backward();

    x.grad().print();
}