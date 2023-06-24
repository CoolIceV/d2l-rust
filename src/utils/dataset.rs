#![cfg_attr(debug_assertions, allow(dead_code))]

use tch::Tensor;
use std::cmp::min;
use rand::seq::SliceRandom;
use tch::IndexOp;

pub fn data_iter(batch_size: usize, features: &Tensor, labels: &Tensor) -> Vec<(Tensor, Tensor)> {
    let mut rng = rand::thread_rng();
    let num_examples = *features.size().get(0).unwrap();
    let mut indices: Vec<i64> = (0..num_examples).collect();
    indices.shuffle(&mut rng);
    let mut iter = Vec::new();
    for i in (0..num_examples).step_by(batch_size) {
        let start = i;
        let end = min(i+batch_size as i64, num_examples);
        iter.push((features.i(start..end), labels.i(start..end)));
    }
    iter
}