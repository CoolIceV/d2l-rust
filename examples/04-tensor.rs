use tch::{Tensor, Kind, Device};

fn main() {

    let t = Tensor::range(2, 13,  (Kind::Uint8, Device::Cpu));
    let t = t * 2;
    t.print();
    println!("{}", t);

    let t = t.reshape([3, 4]);
    println!("{}", t);

    let t = Tensor::ones([3, 5], (Kind::Uint8, Device::Cpu));
    println!("{}", t);

    let x = Tensor::from_slice(&[1.8, 2f64, 4.0, 8f64]);
    let y = Tensor::from_slice(&[2, 2, 2, 2]);
    let z = x+y;
    z.print();
    // let z = x-y;
    // z.print();
    // let z = x*y;
    // z.print();
    // let z = x/y;
    // z.print();
    let x = Tensor::from_slice(&[1.8, 2f64, 4.0, 8f64]);
    let x = x.reshape([2, 2]);
    let y = Tensor::from_slice(&[2, 2, 2, 2]);
    let y = y.reshape([2, 2]);
    let t = Tensor::concat(&[x, y], 0);
    println!("{}", t);
    
    let x = Tensor::from_slice(&[1.8, 2f64, 4.0, 8f64]);
    let x = x.reshape([2, 2]);
    let y = Tensor::from_slice(&[2, 2, 2, 2]);
    let y = y.reshape([2, 2]);
    println!("{}", x == y);
    println!("{}", x.eq_tensor(&y));
    
    let x = Tensor::from_slice(&[1.8, 2f64, 4.0, 8f64]);
    let x = x.reshape([1, 4]);
    let y = Tensor::from_slice(&[2, 3, 4]);
    let y = y.reshape([3, 1]);
    println!("{}", x+y);

    let x = Tensor::range(2, 13,  (Kind::Uint8, Device::Cpu));
    let x = x.reshape([-1, 1]);
    let y = Tensor::range(2, 13,  (Kind::Uint8, Device::Cpu));
    let x = x.reshape([1, -1]);
    println!("{}", x+y);

    let t = Tensor::range(2, 13,  (Kind::Uint8, Device::Cpu));
    println!("{}", t.get(-1));

}