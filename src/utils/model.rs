use tch::Tensor;

pub trait Model {
    fn forward(&mut self, x: &Tensor) -> Tensor;
    fn update(&mut self, batch_size: i64, lr: f64);
    fn set_training(&mut self, training: bool){ }
}