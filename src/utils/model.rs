use tch::Tensor;

pub trait Model {
    fn net(&self, x: &Tensor) -> Tensor;
    fn updater(&mut self, batch_size: i64, lr: f64);
}