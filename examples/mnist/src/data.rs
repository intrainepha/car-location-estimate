use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    tensor::{backend::Backend, Data, ElementConversion, Int, Tensor},
};

#[derive(Clone, Debug)]
pub struct MnistBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> MnistBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<MnistItem, MnistBatch<B>> for MnistBatcher<B> {
    fn batch(&self, items: Vec<MnistItem>) -> MnistBatch<B> {
        let images = items // take items Vec<MnistItem>
            .iter() // create an iterator over it
            .map(|item| Data::<f32, 2>::from(item.image)) // for each item, convert the image to float32 data struct
            .map(|data| Tensor::<B, 2>::from_data(data.convert(), &self.device)) // for each data struct, create a tensor on the device
            .map(|tensor| tensor.reshape([1, 28, 28])) // for each tensor, reshape to the image dimensions [C, H, W]
            .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081) // for each image tensor, apply normalization
            .collect(); // consume the resulting iterator & collect the values into a new vector
        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data(
                    Data::from([(item.label as i64).elem()]),
                    &self.device,
                )
            })
            .collect();
        let images = Tensor::cat(images, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);
        MnistBatch { images, targets }
    }
}
