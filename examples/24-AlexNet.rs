use d2l::utils::*;
use tch::{ Device, nn, vision, Tensor, IndexOp};
use tch::nn::{Module, OptimizerConfig, ModuleT};

fn main() {
    tch::manual_seed(42);

    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);

    let net = AlexNet::new(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, 1e-4).unwrap();
    
    let mut m = vision::mnist::load_dir("data/mnist").unwrap();
    let n = m.test_images.size()[0];
    m.test_images = m.test_images.resize(&[n, 224, 224]);

    let n = m.train_images.size()[0];
    m.train_images = m.train_images.resize(&[n, 224, 224]);
    // println!("{:?}", m.train_images.size());
    // m.train_images.i(0).print();
    for epoch in 1..100 {
        for (bimages, blabels) in m.train_iter(128).shuffle().to_device(vs.device()) {
            let loss = net.forward_t(&bimages, true).cross_entropy_for_logits(&blabels);
            opt.backward_step(&loss);
        }
        let test_accuracy =
            net.batch_accuracy_for_logits(&m.test_images, &m.test_labels, vs.device(), 128);
        println!("epoch: {:4} test acc: {:5.2}%", epoch, 100. * test_accuracy,);
    }
}

#[derive(Debug)]
struct AlexNet {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    conv3: nn::Conv2D,
    conv4: nn::Conv2D,
    conv5: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,
    fc3: nn::Linear,
}

impl AlexNet {
    fn new(vs: &nn::Path) -> Self {
        let mut con_config1 = nn::ConvConfig::default();
        con_config1.padding = 1;
        con_config1.stride = 4;
        let conv1 = nn::conv2d(vs / "conv1", 1, 96, 11, con_config1);
        
        let mut con_config2 = nn::ConvConfig::default();
        con_config2.padding = 2;
        let conv2 = nn::conv2d(vs / "conv2", 96, 256, 5, con_config2);
        
        let mut con_config3 = nn::ConvConfig::default();
        con_config3.padding = 1;
        let conv3 = nn::conv2d(vs / "conv3", 256, 384, 3, con_config3);
        let conv4 = nn::conv2d(vs / "conv4", 384, 384, 3, con_config3);
        let conv5 = nn::conv2d(vs / "conv5", 384, 256, 3, con_config3);

        let fc1 = nn::linear(vs / "fc1", 6400, 4096, Default::default());
        let fc2 = nn::linear(vs / "fc2", 4096, 4096, Default::default());
        let fc3 = nn::linear(vs / "fc3", 4096, 10, Default::default());
        AlexNet { conv1, conv2, conv3, conv4, conv5, fc1, fc2, fc3 }
    }
}

impl nn::ModuleT for AlexNet {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.view([-1, 1, 224, 224])
            .apply(&self.conv1).relu()
            .max_pool2d(3, 2, 0, 1, false)
            .apply(&self.conv2).relu()
            .max_pool2d(3, 2, 0, 1, false)
            .apply(&self.conv3).relu()
            .apply(&self.conv4).relu()
            .apply(&self.conv5).relu()
            .max_pool2d(3, 2, 0, 1, false)
            .flatten(1, -1)
            .apply(&self.fc1).dropout(0.5, train)
            .apply(&self.fc2).dropout(0.5, train)
            .apply(&self.fc3)
    }
}