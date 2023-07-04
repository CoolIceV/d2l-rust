use d2l::utils::*;
use tch::{ Device, nn, vision};
use tch::nn::{Module, OptimizerConfig};

fn main() {
    tch::manual_seed(42);

    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let vs_root = &vs.root();

    let mut con_config1 = nn::ConvConfig::default();
    con_config1.padding = 2;

    let net: nn::Sequential = nn::seq()
        .add_fn(|x| x.view_(&[-1, 1, 28, 28]))
        .add(nn::conv2d(vs_root / "conv1", 1, 6, 5, con_config1))
        .add_fn(|x| x.avg_pool2d_default(2))
        .add(nn::conv2d(vs_root / "conv2", 6, 16, 5, nn::ConvConfig::default()))
        .add_fn(|x| x.sigmoid() )
        .add_fn(|x| x.avg_pool2d_default(2))
        .add_fn(|x| x.flatten(1, -1))
        .add(nn::linear(vs_root / "liner1", 16*5*5, 120, nn::LinearConfig::default()))
        .add_fn(|x| x.sigmoid() )
        .add(nn::linear(vs_root / "liner2", 120, 84, nn::LinearConfig::default()))
        .add(nn::linear(vs_root / "liner3", 84, 10, nn::LinearConfig::default()));

    // let x = Tensor::rand(&[1, 1, 28, 28], (Kind::Float, device));

    let mut m = vision::mnist::load_dir("data/mnist").unwrap();
    m.train_images = m.train_images.to_device(device);
    m.train_labels = m.train_labels.to_device(device);
    m.test_images = m.test_images.to_device(device);
    m.test_labels = m.test_labels.to_device(device);

    let lr = 0.3;
    let num_epochs = 100;
    let batch_size = 0;

    let mut opt: nn::Optimizer = nn::Sgd::default().build(&vs, lr).unwrap();
    
    let mut loss_vec: Vec<f64> = Vec::new();

    for epoch in 1..num_epochs {

        for (x, y) in data_iter(batch_size, &m.train_images, &m.train_labels) {
            let loss = net.forward(&x).cross_entropy_for_logits(&y);
            opt.backward_step(&loss);
                   
        }
        
        let loss = net.forward(&m.train_images).cross_entropy_for_logits(&m.train_labels);
        let train_accuracy = net.forward(&m.train_images).accuracy_for_logits(&m.train_labels);
        let test_accuracy = net.forward(&m.test_images).accuracy_for_logits(&m.test_labels);
        loss_vec.push(f64::try_from(&loss).unwrap());
         println!(
            "epoch: {:4} train loss: {:8.5} train acc: {:5.2}% test acc: {:5.2}%",
            epoch,
            f64::try_from(&loss).unwrap(),
            100. * f64::try_from(&train_accuracy).unwrap(),
            100. * f64::try_from(&test_accuracy).unwrap(),
        );
    }

    let train_iter = data_iter(batch_size, &m.train_images, &m.train_labels);
    let test_iter = data_iter(batch_size, &m.test_images, &m.test_labels);
    train_ch6(&net, &train_iter, &test_iter, num_epochs, &mut opt)

}