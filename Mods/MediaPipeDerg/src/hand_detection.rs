pub const HAND_DETECTION_SIZE: usize = 192;

#[cfg(test)]
mod tests {
    use burn::{
        Tensor,
        backend::NdArray,
        tensor::{Shape, TensorData},
    };
    use image::{ImageReader, Pixel};

    use crate::{hand_detection::HAND_DETECTION_SIZE, models::Model};

    #[test]
    fn hand_detection_test() {
        type OurBackend = NdArray<f32>;

        let model: Model<OurBackend> = Model::default();
        let device = Default::default();

        let hand_image = ImageReader::open(concat!(env!("CARGO_MANIFEST_DIR"), "/hand.jpg"))
            .expect("Image could not be loaded")
            .with_guessed_format()
            .expect("Could not guess image format")
            .decode()
            .expect("Could not decode image");

        let resized = hand_image.resize(
            HAND_DETECTION_SIZE as u32,
            HAND_DETECTION_SIZE as u32,
            image::imageops::FilterType::Triangle,
        ); // Decent result with okay performance for now. Nearest would be much faster.
        assert_eq!(resized.height(), resized.width());

        let concatted_pixels: Vec<f32> = resized
            .to_rgb32f()
            .pixels()
            .flat_map(|pix| pix.channels())
            .copied()
            .collect();

        let tensor = Tensor::<OurBackend, 4>::from_data(
            TensorData::new(
                concatted_pixels,
                Shape::new([1, HAND_DETECTION_SIZE, HAND_DETECTION_SIZE, 3]),
            ),
            &device,
        );
        let max = tensor.clone().max().into_scalar();
        assert!(max <= 1.0);
        let min = tensor.clone().min().into_scalar();
        assert!(min >= 0.0);

        let (result0, result1) = model.forward(tensor);
        println!("Result 1: {result0} Result 2: {result1}");
    }
}
