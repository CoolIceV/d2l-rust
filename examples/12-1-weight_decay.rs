use d2l::utils::*;
use tch::{ Kind, Tensor, Device, nn::Sgd};

struct Linear{
    ws: Tensor,
    bs: Tensor,
}

impl Linear {
    fn new(in_dim: i64, out_dim: i64) -> Linear {
        let ws = Tensor::zeros([in_dim, out_dim], (Kind::Float, Device::cuda_if_available())).normal_(0., 1.).set_requires_grad(true);
        let bs = Tensor::zeros([out_dim], (Kind::Float, Device::cuda_if_available())).set_requires_grad(true);
        Linear{ws, bs}
        // Linear{ws}
    }

}

impl Model for Linear {
    fn forward(&self, x: &Tensor) -> Tensor {
        // x.matmul(&self.ws)
        x.matmul(&self.ws) + &self.bs
    }

    fn update(&mut self, batch_size: i64, lr: f64) {
        let batch_size = batch_size as usize;
        sgd(vec![&mut self.ws, &mut self.bs], lr, batch_size);
        // sgd(vec![&mut self.ws], lr, batch_size);
    }
}

fn main() {
    let n_train = 20;
    let n_test = 100;
    let num_inputs = 200;
    let batch_size = 5;

    let true_w = Tensor::ones([num_inputs, 1], (Kind::Float, Device::cuda_if_available()))*0.01;
    let true_b = Tensor::from_slice(&[0.05]).to_device(Device::cuda_if_available());
    let train_data = synthetic_data(&true_w, &true_b, n_train);
    let test_data = synthetic_data(&true_w, &true_b, n_test);
    let train_iter = data_iter(batch_size, &train_data.0, &train_data.1);
    let test_iter = data_iter(batch_size, &test_data.0, &test_data.1);

    let lambd = 90.;
    let mut model = Linear::new(num_inputs, 1);
    let num_epochs = 100;
    let lr = 0.003;

    let mut a = Animator::new(&["train loss", "test loss"]);

    for epoch in 0..num_epochs {
        for (x, y) in &train_iter {
            let y_hat = model.forward(&x);
            let l: Tensor = squared_loss(&y_hat, &y) + lambd * l2_penalty(&model.ws);
            l.sum(Kind::Float).backward();
            model.update(batch_size as i64, lr);
        }

        let train_loss = evaluate_loss(&mut model, &train_iter, squared_loss);
        let test_loss = evaluate_loss(&mut model, &test_iter, squared_loss);
        println!("epoach: {}, train loss:{}, test_loss:{}", epoch, train_loss, test_loss);
        a.add_point("train loss", epoch as f64, train_loss);
        a.add_point("test loss", epoch as f64, test_loss);
        a.draw();
    }

    println!("w的L2范数是：");
    model.ws.norm().print()
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