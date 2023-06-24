/* Some very simple models trained on the MNIST dataset.
   The 4 following dataset files can be downloaded from http://yann.lecun.com/exdb/mnist/
   These files should be extracted in the 'data/minist' directory.
     train-images-idx3-ubyte
     train-labels-idx1-ubyte
     t10k-images-idx3-ubyte
     t10k-labels-idx1-ubyte
*/

use tch::nn::{Module, OptimizerConfig};
use tch::{nn, vision, Device};

const IMAGE_DIM: i64 = 784;
const LABELS: i64 = 10;

fn main() {
    
    tch::manual_seed(42);
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    
    let cfg = nn::LinearConfig {
        ws_init: nn::Init::Const(0.),
        bs_init: Some(nn::Init::Const(0.)),
        bias: true,
    };

    println!("{:?}", cfg);

    let net = nn::seq()
        .add(nn::linear(
            &vs.root() / "layer1",
            IMAGE_DIM,
            LABELS,
            cfg,
        ));


    let mut m = vision::mnist::load_dir("data/mnist").unwrap();
    m.train_images = m.train_images.to_device(device);
    m.train_labels = m.train_labels.to_device(device);
    m.test_images = m.test_images.to_device(device);
    m.test_labels = m.test_labels.to_device(device);

    let lr = 0.3;
    let num_epochs = 500;
    let mut opt = nn::Sgd::default().build(&vs, lr).unwrap();
    for epoch in 1..num_epochs {
        let loss = net.forward(&m.train_images).cross_entropy_for_logits(&m.train_labels);
        opt.backward_step(&loss);
        let test_accuracy = net.forward(&m.test_images).accuracy_for_logits(&m.test_labels);
        println!(
            "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
            epoch,
            f64::try_from(&loss).unwrap(),
            100. * f64::try_from(&test_accuracy).unwrap(),
        );
    }

}