use tch::{nn, vision, nn::ModuleT, nn::OptimizerConfig, Device};
use d2l::utils::*;

const IMAGE_DIM: i64 = 784;
const HIDDEN_NODES1: i64 = 256;
const HIDDEN_NODES2: i64 = 256;
const LABELS: i64 = 10;

fn main() {
    tch::manual_seed(42);
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);


    let mut m = vision::mnist::load_dir("data/mnist").unwrap();
    m.train_images = m.train_images.to_device(device);
    m.train_labels = m.train_labels.to_device(device);
    m.test_images = m.test_images.to_device(device);
    m.test_labels = m.test_labels.to_device(device);

    let lr = 0.5;
    let num_epochs = 10;
    let batch_size = 256;
    let p = &vs.root();

    let net = nn::seq_t()
        .add(nn::linear(p / "layer1", IMAGE_DIM, HIDDEN_NODES1, Default::default()))
        .add_fn(|x| x.relu())
        .add_fn_t(|xs, train| xs.dropout(0.2, train))
        .add(nn::linear(p / "layer2", HIDDEN_NODES1, HIDDEN_NODES2, Default::default()))
        .add_fn(|x| x.relu())
        .add_fn_t(|xs, train| xs.dropout(0.2, train))
        .add(nn::linear(p / "layer3", HIDDEN_NODES2, LABELS, Default::default()));
    let mut opt = nn::Sgd::default().build(&vs, lr).unwrap();
    
    let mut loss_vec: Vec<f64> = Vec::new();

    for epoch in 1..num_epochs {

        for (x, y) in data_iter(batch_size, &m.train_images, &m.train_labels) {
            let loss = net.forward_t(&x, true).cross_entropy_for_logits(&y);
            opt.backward_step(&loss);
                   
        }

        let loss = net.forward_t(&m.train_images, false).cross_entropy_for_logits(&m.train_labels);
        let test_accuracy = net.forward_t(&m.test_images, false).accuracy_for_logits(&m.test_labels);
        loss_vec.push(f64::try_from(&loss).unwrap());
         println!(
            "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
            epoch,
            f64::try_from(&loss).unwrap(),
            100. * f64::try_from(&test_accuracy).unwrap(),
        );
    }
}