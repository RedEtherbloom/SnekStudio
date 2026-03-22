pub const HAND_DETECTOR_SIZE: usize = 192;

#[cfg(test)]
mod tests {
    use burn::{
        Tensor,
        backend::NdArray,
        prelude::Backend,
        tensor::{Shape, TensorData},
    };
    use image::{ImageReader, Pixel};
    use tracing::{debug, debug_span, info};

    use crate::{hand_detection::HAND_DETECTOR_SIZE, models::hand_detector::Model};

    #[test_log::test]
    fn hand_detector_test() {
        let span = debug_span!("hand_detector_test");
        let _guard = span.enter();
        debug!("Enterred test");
        type OurBackend = NdArray<f32>;

        let model: Model<OurBackend> = Model::default();
        let device = Default::default();
        debug!("Finished device setup");
        OurBackend::seed(&device, 42);
        debug!("Finished seeding");

        let hand_image =
            ImageReader::open(concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/hand.jpg"))
                .expect("Image could not be loaded")
                .with_guessed_format()
                .expect("Could not guess image format")
                .decode()
                .expect("Could not decode image");
        debug!("Loaded hand image");

        let resized = hand_image.resize(
            HAND_DETECTOR_SIZE as u32,
            HAND_DETECTOR_SIZE as u32,
            image::imageops::FilterType::Triangle,
        ); // Decent result with okay performance for now. Nearest would be much faster.
        assert_eq!(resized.height(), resized.width());
        debug!("Resized hand image");

        let concatted_pixels: Vec<f32> = resized
            .to_rgb32f()
            .pixels()
            .flat_map(|pix| pix.channels())
            .copied()
            .collect();
        debug!("Collected pixels");

        let tensor_data = TensorData::new(
            concatted_pixels,
            Shape::new([1, HAND_DETECTOR_SIZE, HAND_DETECTOR_SIZE, 3]),
        );
        debug!("Converted pixels to tensor data");
        let tensor = Tensor::<OurBackend, 4>::from_data(tensor_data, &device);
        debug!("Moved pixel tensor onto device");
        let max = tensor.clone().max().into_scalar();
        assert!(max <= 1.0);
        let min = tensor.clone().min().into_scalar();
        assert!(min >= 0.0);
        debug!("Finished data sanity check");

        let (result1, result2) = model.forward(tensor);
        debug!(
            result_tensor_1 = ?result1,
            result_tensor_2 = ?result2,
            "Finished inference",
        );
        info!("Test succeeded without issues");
    }
}
