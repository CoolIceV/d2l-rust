use tch::Tensor;

pub fn sgd(params: Vec<&mut Tensor>, lr: f64, batch_size: usize) {
    tch::no_grad(|| {
        for param in params.into_iter() {
            *param -= lr * param.grad() / (batch_size as f64);
            _ = param.grad().zero_();
        } 
    })
}