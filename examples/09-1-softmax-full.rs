/* Some very simple models trained on the MNIST dataset.
   The 4 following dataset files can be downloaded from http://yann.lecun.com/exdb/mnist/
   These files should be extracted in the 'data/minist' directory.
     train-images-idx3-ubyte
     train-labels-idx1-ubyte
     t10k-images-idx3-ubyte
     t10k-labels-idx1-ubyte
*/

use tch::{vision, Kind, Tensor, Device, IndexOp};

const IMAGE_DIM: i64 = 784;
const LABELS: i64 = 10;

fn main() {
    let device = Device::cuda_if_available();
    let mut m = vision::mnist::load_dir("data/mnist").unwrap();
    m.train_images = m.train_images.to_device(device);
    m.train_labels = m.train_labels.to_device(device);
    m.test_images = m.test_images.to_device(device);
    m.test_labels = m.test_labels.to_device(device);
    // println!("{:?}", m.train_images.size());
    // println!("{:?}", m.train_labels.size());
    // println!("{:?}", m.test_images.size());
    // println!("{:?}", m.train_images.i(0).size());
    // println!("{:?}", m.test_labels.size());

    let mut ws = Tensor::zeros([IMAGE_DIM, LABELS], (Kind::Float, device)).set_requires_grad(true);
    let mut bs = Tensor::zeros([LABELS], (Kind::Float, device)).set_requires_grad(true);

    // let x = Tensor::range(1, 10, kind::FLOAT_CPU);
    // let x = x.reshape([2, 5]);
    // let x_prob = softmax(&x);
    // x_prob.print();
    // x_prob.sum(Kind::Float).print();

    // let y = Tensor::from_slice(&[0, 2]);
    // let y_hat = Tensor::from_slice(&[0.1, 0.3, 0.6, 0.3, 0.2, 0.5]).reshape([-1, 3]);
    // // y_hat.index(&[Some(Tensor::from_slice(&[0, 1])), Some(y)]).print();
    // cross_entropy(&y_hat, &y).print();

    // println!("{}", accuracy(&y_hat, &y) / y.size()[0] as f64);

    let lr = 0.3;
    let num_epochs = 1000;
    let batch_size = m.train_images.size()[0] as usize;
    let net = linear_net;
    let loss = cross_entropy;

    for epoch in 0..num_epochs {
        let l = loss(&net(&m.train_images, &ws, &bs), &m.train_labels);
        l.sum(Kind::Float).backward();
        sgd(vec![&mut ws, &mut bs], lr, batch_size);

        tch::no_grad(||{
            let y_hat = net(&m.test_images, &ws, &bs);
            let loss = loss(&y_hat, &m.test_labels);
            let accuracy = accuracy(&y_hat, &m.test_labels);

            println!(
                "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
                epoch,
                loss.mean(Kind::Float).double_value(&[]),
                100. * accuracy,
            );
        })
       
    }
}

fn softmax(x: &Tensor) -> Tensor {
    let x_exp = x.exp();
    let partition = x_exp.sum_dim_intlist(1, true, Kind::Float);
    x_exp / partition
}

fn linear_net(x: &Tensor, ws: &Tensor, bs: &Tensor) -> Tensor {
    softmax(&(x.reshape([-1, ws.size()[0]]).matmul(&ws) + bs))
}

fn cross_entropy(y_hat: &Tensor, y: &Tensor) -> Tensor {
    let x = Tensor::arange(y_hat.size()[0], (Kind::Int64, y.device()));
    -y_hat.index(&[Some(&x), Some(y)]).log()
}

fn accuracy(y_hat: &Tensor, y: &Tensor) -> f64 {
    let y_hat = if y_hat.size().len() > 1 && y_hat.size()[1] > 1 {
        y_hat.argmax(1, false)
    } else {
        y_hat.i(..)
    };

    let cmp = y_hat.eq_tensor(y);
    cmp.to_kind(y.kind()).sum(Kind::Float).double_value(&[]) / y.size()[0] as f64
}

fn sgd(params: Vec<&mut Tensor>, lr: f64, batch_size: usize) {
    tch::no_grad(|| {
        for param in params.into_iter() {
            *param -= lr * param.grad() / (batch_size as f64);
            _ = param.grad().zero_();
        } 
    })
}