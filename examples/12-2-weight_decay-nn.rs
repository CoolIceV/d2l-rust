use tch::nn::{Module, OptimizerConfig};
use tch::{nn, Device, Kind, Reduction, Tensor};
use std::ops::Add;
use d2l::utils::*;
fn main() {
    let n_train = 20;
    let n_test = 100;
    let num_inputs = 200;
    let batch_size = 5;

    let true_w = Tensor::ones([num_inputs, 1], (Kind::Float, Device::cuda_if_available()))*0.01;
    let true_b = Tensor::from_slice(&[0.05f32]).to_device(Device::cuda_if_available());
    let train_data = synthetic_data(&true_w, &true_b, n_train);
    let test_data = synthetic_data(&true_w, &true_b, n_test);
    let train_iter = data_iter(batch_size, &train_data.0, &train_data.1);
    let test_iter = data_iter(batch_size, &test_data.0, &test_data.1);

    let lambd = 90.;

    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    
    let cfg = nn::LinearConfig {
        ws_init: nn::Init::Const(0.),
        bs_init: Some(nn::Init::Const(0.)),
        bias: true,
    };
    let model = nn::seq()
        .add(nn::linear(
            &vs.root() / "layer1",
            num_inputs,
            1,
            cfg,
        ));
    
    let num_epochs = 100;
    let lr = 0.003;

    let mut opt = nn::Sgd::default().build(&vs, lr).unwrap();
    opt.set_weight_decay(lambd);

    let mut a = Animator::new(&["train loss", "test loss"]);

    for epoch in 0..num_epochs {
        for (x, y) in &train_iter {
            let loss = model
                .forward(&x)
                .mse_loss(&y, Reduction::Mean);
            opt.backward_step(&loss);
        }

        // let train_loss = evaluate_loss(&mut model, &train_iter, squared_loss);
        // let test_loss = evaluate_loss(&mut model, &test_iter, squared_loss);
        // println!("epoach: {}, train loss:{}, test_loss:{}", epoch, train_loss, test_loss);
        // a.add_point("train loss", epoch as f64, train_loss);
        // a.add_point("test loss", epoch as f64, test_loss);
        // a.draw();
    }

    // println!("w的L2范数是：");
    // println!("{:?}", model.get(0).unwrap().ws().unwrap().norm(Kind::Float));
}

fn synthetic_data(w: &Tensor, b: &Tensor, n: i64) -> (Tensor, Tensor) {
    let x = Tensor::randn([n, w.size()[0]], (Kind::Float, Device::cuda_if_available()));
    let y = x.matmul(w) + b;
    let size = y.size();
    let y = y + Tensor::zeros(size, (Kind::Float, Device::cuda_if_available())).normal_(0., 0.1);
    (x, y)
}

fn l2_penalty(w: &Tensor) -> Tensor {
    w.pow(&Tensor::from(2)).sum(Kind::Float) / 2
}