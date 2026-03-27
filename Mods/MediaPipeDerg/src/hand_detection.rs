use burn::config::Config;

pub const HAND_DETECTOR_SIZE: usize = 192;

pub enum BoundingBoxType {
    XYWH,   // X1, Y1, Width, Height
    YXHW,   // Y1, X1, Height, Width
    XYXY,   // X1, Y1, X2, Y2
    CXCYWH, // X_Center, Y_Center, Width, Height
    Unknown,
}

#[derive(Config, Debug)]
pub struct DetectionConfig {
    #[config(default = 0)]
    offset_box: usize, // Offset of box data in row. Usually starts at beginning of row.
    #[config(default = 4)]
    number_values_per_box: usize,

    offset_keypoints: usize, // Offset to keypoint data in tensor
    number_keypoints: usize,
    number_values_per_keypoint: usize, // Number of predicted values per keypoint

    #[config(default = 0.5)]
    nms_iou_min_threshold: f32, // Minimum overlap ratio for boxes to be considered duplicate
    #[config(default = 0.5)]
    nms_score_min_threshold: f32, // Minimum confidence for detected bounding box to be considered
    // valid

    // Not yet implemented options
    #[config(default = 0)]
    number_rows: usize, // Number of rows to process, or 0 for all
    #[config(default = 1)]
    number_classes: usize, // Unused for now
    #[config(default = 0)]
    summed_row_size: usize, // Total summed size of a single detection tensor row. Mostly useful for validation.
    #[config(default = false)]
    sigmoid_score: bool, // Apply sigmoid function to scores. Used by upstream Mediapipe
}

pub fn default_hand_detection_config() -> DetectionConfig {
    DetectionConfig::new(4, 2, 7)
}

#[cfg(test)]
mod tests {
    use burn::{
        Tensor,
        backend::NdArray,
        prelude::Backend,
        tensor::{Shape, TensorData, s},
    };
    use burn_vision::{Nms, NmsOptions};
    use image::{ImageReader, Pixel};
    use tracing::{debug, debug_span, info};

    use crate::{
        hand_detection::{HAND_DETECTOR_SIZE, default_hand_detection_config},
        models::hand_detector::Model,
    };

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

        // TODO: Can be done on the GPU using burn_vision::Transform
        let resized = hand_image.resize(
            HAND_DETECTOR_SIZE as u32,
            HAND_DETECTOR_SIZE as u32,
            image::imageops::FilterType::Triangle,
        ); // Decent result with okay performance for now. Nearest would be much faster.
        assert_eq!(resized.height(), resized.width());
        debug!("Resized hand image");

        let concatted_pixels: Vec<f32> = resized.clone()
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

        let (box_tensor, score_tensor) = model.forward(tensor);
        debug!(?box_tensor, ?score_tensor, "Finished inference");

        debug!("Beginning Non-Maximum Suppression");
        let detection_config = default_hand_detection_config();
        let boxes_subslice = box_tensor.clone().squeeze().slice(s![
            ..,
            detection_config.offset_box..detection_config.number_values_per_box,
        ]);
        let nms_options = NmsOptions {
            iou_threshold: detection_config.nms_iou_min_threshold,
            score_threshold: detection_config.nms_score_min_threshold,
            max_output_boxes: 0,
        };
        // TODO: This gives incorrect results as mediapipe uses CXCYWH afaik, while
        // burn_vision::Nms uses XYXY.
        let box_indices = boxes_subslice.nms(score_tensor.clone().squeeze(), nms_options);
        let remaining_boxes = box_tensor.select(1, box_indices.clone());
        debug!(
            ?box_indices,
            ?remaining_boxes,
            "Finished Non-Maxium Suppression"
        );
        let out:Vec<f32> = remaining_boxes.iter_dim(1).map(|el| el.squeeze_dims::<1>(&[0, 1])).next().expect("Expected to get at least one box").into_data().to_vec().unwrap();
        assert!(out.len() >= 4);
        let x_center = out[0];
        let y_center = out[1];
        let w = out[2];
        let h = out[3];

        let with_drawn_box = imageproc::drawing::draw_hollow_rect(&resized, imageproc::rect::Rect::at((x_center - w / 2.0) as i32, (y_center - h/2.0) as i32).of_size(w as u32, h as u32), image::Rgba::<u8>([127, 0, 127, 0]));
        with_drawn_box.save("test_with_box.png").expect("Could not save image.");
        debug!("Successfully saved image");

        info!("Test succeeded without issues");
    }
}
