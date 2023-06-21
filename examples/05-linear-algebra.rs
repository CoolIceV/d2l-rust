use tch::{Tensor, Kind, Device};

fn main() {
    let x = Tensor::from_slice(&[1.8, 2f64, 4.0, 8f64]);
    let x = x.reshape([2, 2]);
    x.print();
    x.t_copy().print();

    x.eq_tensor(&x.t_copy()).print();

    let t = Tensor::range(2, 13,  (Kind::Float, Device::cuda_if_available()));
    t.reshape([2, 2, 3]).print();

    t.sum(Kind::Float).print();

    let t = t.reshape([2, 2, 3]);
    t.sum_dim_intlist(1, true, Kind::Float).print();
    t.sum_dim_intlist(1, false, Kind::Float).print();

    t.mean(Kind::Float).print();

    t.print();
    t.cumsum(0, Kind::Float).print();

    let x = Tensor::range(2, 13,  (Kind::Float, Device::cuda_if_available()));
    let x = x.reshape([-1, 3]);
    x.print();
    let y = Tensor::range(2, 4,  (Kind::Float, Device::cuda_if_available()));
    y.print();
    x.mv(&y).print();

    let x = Tensor::range(2, 13,  (Kind::Float, Device::cuda_if_available()));
    let x = x.reshape([-1, 3]);
    let y = Tensor::range(2, 13,  (Kind::Float, Device::cuda_if_available()));
    let y = y.reshape([-1, 3]);
    x.mm(&y.t_copy()).print();

    x.abs().sum(Kind::Float).print();
    let x = x.totype(Kind::Float);
    x.norm().print();
}