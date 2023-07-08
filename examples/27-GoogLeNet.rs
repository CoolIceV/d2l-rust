use d2l::utils::*;
use tch::{ Device, nn, vision, Tensor, IndexOp, Kind};
use tch::nn::{Module, OptimizerConfig, ModuleT};

fn main() {
    tch::manual_seed(42);

    let device = Device::cuda_if_available();
    let vars = nn::VarStore::new(device);
    let vs  = &vars.root();
    let x = Tensor::rand(&[1, 1, 224, 224], (Kind::Float, device));
    let b1 = nn::seq_t().add(nn::conv2d(vs, 1, 64, 7, nn::ConvConfig{padding: 1, stride: 2, ..Default::default()}))
            .add_fn(|x| x.max_pool2d(3, 2, 1, 1, false));
    
    let b2 = nn::seq_t().add(nn::conv2d(vs, 64, 64, 1, Default::default()))
            .add(nn::conv2d(vs, 64, 192, 3, nn::ConvConfig{padding: 1, ..Default::default()}))
            .add_fn(|x| x.max_pool2d(3, 2, 1, 1, false));
    let b3 = nn::seq_t()
            .add(inception(vs, 192, 64, [96, 128], [16, 32], 32))
            .add(inception(vs, 256, 128, [128, 192], [32, 96], 64))
            .add_fn(|x| x.max_pool2d(3, 2, 1, 1, false));
    
    let net = b1.add(b2).add(b3);
    
    net.forward_t(&x, true).print();
}

fn inception(vs: &nn::Path, in_channels: i64, c1: i64, c2: [i64;2], c3: [i64;2], c4: i64) -> impl ModuleT {
    let p1_1 = nn::conv2d(vs, in_channels, c1, 1 ,Default::default());
    let p2_1 = nn::conv2d(vs, in_channels, c2[0], 1, Default::default());
    let p2_2 = nn::conv2d(vs, c2[0], c2[1], 3, nn::ConvConfig{padding: 1, ..Default::default()});
    let p3_1 = nn::conv2d(vs, in_channels, c3[0], 1, Default::default());
    let p3_2 = nn::conv2d(vs,  c3[0], c3[1], 5, nn::ConvConfig{padding: 2, ..Default::default()});
    // let p4_1 = nn::conv2d(vs, in_channels, c4, 1, Default::default());
    let p4_2 = nn::conv2d(vs, in_channels, c4, 1, Default::default());
    
    nn::func_t(move |xs, train| {
        let p1 = xs.apply(&p1_1).relu();
        let p2 = xs.apply(&p2_1).apply(&p2_2).relu();
        let p3 = xs.apply(&p3_1).apply(&p3_2).relu();
        let p4 = xs.max_pool2d(3, 1, 1, 1, false).apply(&p4_2).relu();
        Tensor::cat(&[p1, p2, p3, p4], 1)
    })
}