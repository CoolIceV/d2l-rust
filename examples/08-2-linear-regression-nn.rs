
use tch::nn::{Module, OptimizerConfig};
use tch::{nn, Device, Kind, Reduction, Tensor};
use std::ops::Add;
use d2l::utils::dataset::data_iter;

fn synthetic_data(w: Vec<f32>, b: f32, num: i32) -> (Tensor, Tensor){
    let mut x = Tensor::ones([num as i64, w.len() as i64], (Kind::Float, Device::Cpu));
    let x = x.normal_(0.0, 1.0);

    let mut w = Tensor::from_slice(&w);
    let y = x.matmul(&w.t_()).add(Tensor::from(b));
    let mut noize = Tensor::ones(y.size(), (Kind::Float, Device::Cpu));

    let y = y+ noize.normal_(0.0, 0.01); 
    let y = y.reshape([-1, 1]);
    (x, y)
}

fn main() {

    tch::manual_seed(42);
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    
    let cfg = nn::LinearConfig {
        ws_init: nn::Init::Const(0.),
        bs_init: Some(nn::Init::Const(0.)),
        bias: true,
    };


    let net = nn::seq()
        .add(nn::linear(
            &vs.root() / "layer1",
            3,
            1,
            cfg,
        ));

    let (features, labels) = synthetic_data(vec![2.0, -5.4, 2.0], 4.2, 1000);
    let lr = 0.03;
    let num_epochs = 3;
    let batch_size = 10;
    let mut opt = nn::Sgd::default().build(&vs, lr).unwrap();

    println!("{:?}", net);

    for epoch in 0..num_epochs {
        for (x, y) in data_iter(batch_size, &features, &labels) {
            let loss = net
                .forward(&x)
                .mse_loss(&y, Reduction::Mean);
            opt.backward_step(&loss);
            
        }

        tch::no_grad(||{
            let loss = net
                .forward(&features)
                .mse_loss(&labels, Reduction::Mean);

            println!("epoch: {}, loss: {}", epoch, loss)
        });
        
        // TODO: 无法查看某一层
        // println!("======================");
        // println!("{:?}", net);
    }
}