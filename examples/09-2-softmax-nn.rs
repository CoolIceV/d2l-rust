/* Some very simple models trained on the MNIST dataset.
   The 4 following dataset files can be downloaded from http://yann.lecun.com/exdb/mnist/
   These files should be extracted in the 'data/minist' directory.
     train-images-idx3-ubyte
     train-labels-idx1-ubyte
     t10k-images-idx3-ubyte
     t10k-labels-idx1-ubyte
*/

use tch::nn::{Module, OptimizerConfig};
use tch::{nn, vision, Device, Kind};
use d2l::utils::dataset::data_iter;
use plotters::prelude::*;

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

    let net = nn::seq()
        .add(nn::linear(
            &vs.root() / "layer1",
            IMAGE_DIM,
            LABELS,
            cfg,
        ))
        .add_fn(|x| x.softmax(1, Kind::Float));


    let mut m = vision::mnist::load_dir("data/mnist").unwrap();
    m.train_images = m.train_images.to_device(device);
    m.train_labels = m.train_labels.to_device(device);
    m.test_images = m.test_images.to_device(device);
    m.test_labels = m.test_labels.to_device(device);

    let lr = 0.3;
    let num_epochs = 1000;
    let batch_size = 500;

    let mut opt = nn::Sgd::default().build(&vs, lr).unwrap();
    
    let mut loss_vec: Vec<f64> = Vec::new();

    for epoch in 1..num_epochs {

        for (x, y) in data_iter(batch_size, &m.train_images, &m.train_labels) {
            let loss = net.forward(&x).cross_entropy_for_logits(&y);
            opt.backward_step(&loss);
                   
        }

        let loss = net.forward(&m.train_images).cross_entropy_for_logits(&m.train_labels);
        let test_accuracy = net.forward(&m.test_images).accuracy_for_logits(&m.test_labels);
        loss_vec.push(f64::try_from(&loss).unwrap());
         println!(
            "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
            epoch,
            f64::try_from(&loss).unwrap(),
            100. * f64::try_from(&test_accuracy).unwrap(),
        );
    }

    let x = 0..loss_vec.len();

    let root = BitMapBackend::new("data/softmax-nn-loss.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .caption("loss", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f32..(loss_vec.len() as f32), 0f32..0.6).unwrap();

    chart.configure_mesh().draw().unwrap();

    chart
        .draw_series(LineSeries::new(
            x.map(|x| (x as f32, loss_vec[x] as f32)),
            &RED,
        )).unwrap()
        .label("loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw().unwrap();

    root.present().unwrap();

}