use d2l::utils::*;
use tch::{ Device, nn, vision, Tensor, IndexOp};
use tch::nn::{Module, OptimizerConfig, ModuleT};


fn main() {
    // let conv_arch = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)];
    let conv_arch = [(1, 64 / 4), (1, 128 / 4), (2, 256 / 4), (2, 512 / 4), (2, 512 / 4)];
    let vs = nn::VarStore::new(Device::cuda_if_available());

    let net = vgg(&vs.root(), &conv_arch);
    let mut opt = nn::Adam::default().build(&vs, 0.05).unwrap();

    let mut m = vision::mnist::load_dir("data/mnist").unwrap();
    let n = m.test_images.size()[0];
    m.test_images = m.test_images.resize(&[n, 224, 224]);

    let n = m.train_images.size()[0];
    m.train_images = m.train_images.resize(&[n, 224, 224]);

    for epoch in 1..10 {
        for (bimages, blabels) in m.train_iter(128).shuffle().to_device(vs.device()) {
            let loss = net.forward_t(&bimages, true).cross_entropy_for_logits(&blabels);
            opt.backward_step(&loss);
        }
        let test_accuracy =
            net.batch_accuracy_for_logits(&m.test_images, &m.test_labels, vs.device(), 128);
        println!("epoch: {:4} test acc: {:5.2}%", epoch, 100. * test_accuracy);
    }
}

fn vgg(vs: &nn::Path, conv_arch: &[(i64, i64)]) -> impl ModuleT {
    let mut conv_blks = nn::seq_t().add_fn(|x| x.view([-1, 1, 224, 224]));
    let mut in_channels = 1;
    for (i, (num_convs, out_channels)) in conv_arch.iter().enumerate() {
        conv_blks = conv_blks.add(vgg_block(&(vs / i.to_string()), *num_convs, in_channels, *out_channels));
        in_channels = *out_channels;
    }

    let vgg = conv_blks.add_fn(|xs| xs.flatten(1, -1))
                        .add(nn::linear(vs / "fc1",  in_channels * 7 * 7, 4096, Default::default())).add_fn(|xs| xs.relu())
                        .add_fn_t(|x, train| x.dropout(0.5, train))
                        .add(nn::linear(vs / "fc2", 4096, 4096, Default::default())).add_fn(|xs| xs.relu())
                        .add_fn_t(|x, train| x.dropout(0.5, train))
                        .add(nn::linear(vs/ "fc3", 4096, 10, Default::default()));

    vgg
}

fn vgg_block(vs: &nn::Path, num_convs: i64, mut in_channels: i64, mut out_channels: i64) -> impl ModuleT {
    let mut block = nn::seq_t();
    
    let mut conv_config: nn::ConvConfigND<i64> = Default::default();
    conv_config.padding = 1;
    for i in 0..num_convs {
        let mut conv2d = nn::conv2d(vs / i.to_string(), in_channels, out_channels, 3, conv_config);
        block = block.add(conv2d).add_fn(|x| x.relu());
        in_channels = out_channels;
    }
    block = block.add_fn(|xs| xs.max_pool2d_default(2));
    
    block
}