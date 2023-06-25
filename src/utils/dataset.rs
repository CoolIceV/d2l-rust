#![cfg_attr(debug_assertions, allow(dead_code, unused_must_use))]

use tch::{Tensor, vision};
use std::cmp::min;
use rand::seq::SliceRandom;
use tch::IndexOp;
use vision::dataset::Dataset;


pub fn data_iter(batch_size: usize, features: &Tensor, labels: &Tensor) -> Vec<(Tensor, Tensor)> {
    let mut batch_size = batch_size;
    
    if batch_size == 0 {
        batch_size = features.size()[0] as usize
    }
    
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

pub trait ToDevice {
    fn to_device(self, device: tch::Device) -> Self;
}

impl ToDevice for Dataset {
    fn to_device(self, device: tch::Device) -> Self {
        let mut m = Dataset::from(self);
        m.test_images = m.test_images.to_device(device);
        m.test_labels = m.test_labels.to_device(device);
        m.train_images = m.train_images.to_device(device);
        m.train_labels = m.train_labels.to_device(device);
        m
    }
}
