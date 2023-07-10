use tch::{Tensor, nn, Device, Kind};

fn main() {
    let device = Device::cuda_if_available();
    let vars = nn::VarStore::new(device);
    let vs  = &vars.root();
    let blk = Residual::new(vs, 3, 3, false, 1);

    let x= Tensor::rand(&[4, 3, 6, 6], (Kind::Float, device));
    println!("{:?}", blk.forward(&x, true).size()); 

    let blk = Residual::new(vs, 3, 6, true, 2);

    let x= Tensor::rand(&[4, 3, 6, 6], (Kind::Float, device));
    println!("{:?}", blk.forward(&x, true).size()); 
}

struct Residual {
    input_channels: i64,
    num_channels: i64,
    use_1x1conv: bool,
    // strides: i64,
    conv1: nn::Conv<[i64; 2]>,
    conv2: nn::Conv<[i64; 2]>,
    conv3: nn::Conv<[i64; 2]>,
    bn1:  nn::BatchNorm,
    bn2:  nn::BatchNorm,
}

impl Residual {
    fn new(vs: &nn::Path, input_channels: i64, num_channels: i64, use_1x1conv: bool, strides: i64) -> Self {
        let conv1: nn::Conv<_> = nn::conv2d(vs, input_channels, num_channels, 3, nn::ConvConfig{ padding: 1, stride: strides, ..Default::default()});
        let conv2: nn::Conv<[i64; 2]> = nn::conv2d(vs, num_channels, num_channels, 3, nn::ConvConfig{ padding: 1, ..Default::default()});
        let conv3: nn::Conv<[i64; 2]> = nn::conv2d(vs, input_channels, num_channels, 1, nn::ConvConfig{ stride: strides, ..Default::default()});
        let bn1 = nn::batch_norm2d(vs, num_channels, Default::default());
        let bn2: nn::BatchNorm = nn::batch_norm2d(vs, num_channels, Default::default());
        Residual { input_channels, num_channels, use_1x1conv, conv1, conv2, conv3, bn1, bn2 }
    }

    fn forward(&self, x: &Tensor, train: bool) -> Tensor {
    
        let y = x.apply(&self.conv1)
                        .apply_t(&self.bn1, train)
                        .relu();
        let y = y.apply(&self.conv2).apply_t(&self.bn2, train);
        let mut x = x.copy(); 
        if self.use_1x1conv {
            x = x.apply(&self.conv3);
        }
        (y + x).relu()
    }
}

    // fn resnet_block(vs: &nn::Path, input_channels: i64, num_channels: i64, num_residuals: i64, first_block: bool) -> nn::Sequential {
    //     let mut blk = nn::seq_t();
    //     for i in 0..num_residuals {
    //         blk.add(Residual::new(vs, input_channels, num_channels, !first_block && i == 0, 2));
    //     }
    //     blk
    // }