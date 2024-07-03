use burn::backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu};
use burn::optim::AdamConfig;
use mnist::model::ModelConfig;
use mnist::training::{train, TrainingConfig};

fn main() {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device = burn::backend::wgpu::WgpuDevice::default();
    train::<MyAutodiffBackend>(
        "./artifact/",
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device,
    );
}
