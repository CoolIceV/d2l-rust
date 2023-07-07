use d2l::utils::*;
use tch::{ Device, nn, vision, Tensor, IndexOp};
use tch::nn::{Module, OptimizerConfig, ModuleT};

fn main() {
    tch::manual_seed(42);

    let device = Device::cuda_if_available();
    let vars = nn::VarStore::new(device);
    let vs  = &vars.root();
   
    let mut net = nn::seq_t()
        .add_fn(|x| x.view([-1, 1, 224, 224]))
        .add(nin_block(vs, 1, 96, 11, 4, 0))
        .add_fn(|x| x.max_pool2d(3, 2, 0, 1, false))
        .add(nin_block(vs, 96, 256, 5, 1, 2))
        .add_fn(|x| x.max_pool2d(3, 2, 0, 1, false))
        .add(nin_block(vs, 256, 384, 3, 1, 1))
        .add_fn(|x| x.max_pool2d(3, 2, 0, 1, false))
        .add_fn_t(|x, train| x.dropout(0.5, train))
        .add(nin_block(vs, 384, 10, 3, 1, 1))
        .add_fn(|x| x.adaptive_avg_pool2d([1, 1]))
        .add_fn(|x| x.flat_view());
    
    let mut opt = nn::Sgd::default().build(&vars, 0.03).unwrap();

    let mut m = vision::mnist::load_dir("data/mnist").unwrap();
    let n = m.test_images.size()[0];
    m.test_images = m.test_images.resize(&[n, 224, 224]);
    
    let n = m.train_images.size()[0];
    m.train_images = m.train_images.resize(&[n, 224, 224]);

    for epoch in 1..10 {
        for (bimages, blabels) in m.train_iter(128).to_device(vs.device()) {
            let loss = net.forward_t(&bimages, true).cross_entropy_for_logits(&blabels);
            opt.backward_step(&loss);
            println!("{}", loss)
        }
        let test_accuracy =
            net.batch_accuracy_for_logits(&m.train_images, &m.train_labels, vs.device(), 1000);
        println!("epoch: {:4} test acc: {:5.2}%", epoch, 100. * test_accuracy);
    }

    // let lr = 0.3;
    // let num_epochs = 100;
    // let batch_size = 0;

    // let train_iter = data_iter(128, &m.train_images, &m.train_labels);
    // let test_iter = data_iter(512, &m.test_images, &m.test_labels);
    // train_ch6_t(&net, &train_iter, &test_iter, num_epochs, &mut opt)



}

fn nin_block(vs: &nn::Path, in_channels: i64, out_channels: i64, kernel_size: i64, strides: i64, padding: i64) -> impl ModuleT {
    let mut block = nn::seq_t();
    let mut conv_config: nn::ConvConfigND<i64> = Default::default();
    conv_config.stride = strides;
    conv_config.padding = padding;
    
    let mut conv2d = nn::conv2d(vs, in_channels, out_channels, kernel_size, conv_config);
    block = block.add(conv2d).add_fn(|x| x.relu());
    let mut conv2d = nn::conv2d(vs, out_channels, out_channels, 1, Default::default());
    block = block.add(conv2d).add_fn(|x| x.relu());
    let mut conv2d = nn::conv2d(vs, out_channels, out_channels, 1, Default::default());
    block = block.add(conv2d).add_fn(|x| x.relu());
    block
}