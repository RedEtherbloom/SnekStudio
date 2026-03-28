use burn::{Tensor, config::Config, prelude::Backend, tensor::Float};
use tracing::error;

pub const HAND_DETECTOR_SIZE: usize = 192;
// TODO: Write unit tests for the matrix conversions
// Coordinate transformation matrices. These transform the different formats to specify a bounding
// box into each other.
// Corresponds to: [X1, Y1, X2, Y2] => [(X1 + (X2 - X1)), (Y1 + (Y2 - Y1)), (X2 - X1), (Y2 - Y1)]
const BOUNDING_BOX_XYXY_CXCYWH_TRANSFORMATION_MATRIX: [[f32; 4]; 4] = [
    [0.5, 0.0, -1.0, 0.0],
    [0.0, 0.5, 0.0, -1.0],
    [0.5, 0.0, 1.0, 0.0],
    [0.0, 0.5, 0.0, 1.0],
];
// Corresponds to: [CX, CY, W, H] => [(CX - W / 2), (CY - H / 2), (CX + W / 2), (CY + H / 2)]
const BOUNDING_BOX_CXCYWH_XYXY_TRANSFORMATION_MATRIX: [[f32; 4]; 4] = [
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [-0.5, 0.0, 0.5, 0.0],
    [0.0, -0.5, 0.0, 0.5],
];

#[derive(Debug, Copy, Clone)]
pub enum BoundingBoxType {
    XYWH,   // X1, Y1, Width, Height
    YXHW,   // Y1, X1, Height, Width
    XYXY,   // X1, Y1, X2, Y2
    CXCYWH, // X_Center, Y_Center, Width, Height
}

#[derive(Debug, Clone)]
pub struct BoundingBoxes<B: Backend> {
    bounding_type: BoundingBoxType,
    tensor: Tensor<B, 2, Float>,
}

impl<B: Backend> BoundingBoxes<B> {
    // TODO: Proper error type using e.g. color-eyre
    #[allow(clippy::result_unit_err)]
    pub fn new(bounding_type: BoundingBoxType, tensor: Tensor<B, 2, Float>) -> Result<Self, ()> {
        match tensor.shape().dims() {
            [_, 4] => Ok(BoundingBoxes {
                bounding_type,
                tensor,
            }),
            _ => {
                error!("Bounding box slice must be of shape [N, 4]");
                Err(())
            }
        }
    }

    pub fn change_bounding_type(
        old_bounding_type: BoundingBoxType,
        new_bounding_type: BoundingBoxType,
        tensor: Tensor<B, 2, Float>,
    ) -> Tensor<B, 2, Float> {
        match old_bounding_type {
            BoundingBoxType::XYWH => match new_bounding_type {
                BoundingBoxType::XYWH => tensor,
                BoundingBoxType::YXHW => tensor.clone().select(1, [1, 0, 3, 2].into()),
                _ => todo!(),
            },
            BoundingBoxType::YXHW => match new_bounding_type {
                BoundingBoxType::YXHW => tensor,
                BoundingBoxType::XYWH => tensor.clone().select(1, [1, 0, 3, 2].into()),
                _ => todo!(),
            },
            BoundingBoxType::XYXY => match new_bounding_type {
                BoundingBoxType::XYXY => tensor,
                BoundingBoxType::CXCYWH => tensor.clone().matmul(Tensor::from_floats(
                    BOUNDING_BOX_XYXY_CXCYWH_TRANSFORMATION_MATRIX,
                    &tensor.device(),
                )),
                _ => todo!(),
            },
            BoundingBoxType::CXCYWH => match new_bounding_type {
                BoundingBoxType::CXCYWH => tensor,
                BoundingBoxType::XYXY => tensor.clone().matmul(Tensor::from_floats(
                    BOUNDING_BOX_CXCYWH_XYXY_TRANSFORMATION_MATRIX,
                    &tensor.device(),
                )),
                _ => todo!(),
            },
        }
    }

    pub fn change_own_bounding_type(&mut self, new_bounding_type: BoundingBoxType) {
        self.tensor =
            Self::change_bounding_type(self.bounding_type, new_bounding_type, self.tensor.clone());
    }
}

impl<B: Backend> From<BoundingBoxes<B>> for Tensor<B, 2, Float>{
    fn from(value: BoundingBoxes<B>) -> Self {
        value.tensor
    }
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
        hand_detection::{
            BoundingBoxType, BoundingBoxes, HAND_DETECTOR_SIZE, default_hand_detection_config,
        },
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

        let concatted_pixels: Vec<f32> = resized
            .clone()
            .to_rgb32f()
            .pixels()
            .flat_map(|pix| pix.channels())
            .copied()
            .collect();
        debug!("Collected pixels");

        // TODO: Likely produces wrong output format. May be the culprit
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
        let nms_options = NmsOptions {
            iou_threshold: detection_config.nms_iou_min_threshold,
            score_threshold: detection_config.nms_score_min_threshold,
            max_output_boxes: 0,
        };
        let boxes_subslice = box_tensor.clone().squeeze().slice(s![
            ..,
            detection_config.offset_box..detection_config.number_values_per_box,
        ]);
        let mut bounding_boxes =
            BoundingBoxes::<OurBackend>::new(BoundingBoxType::CXCYWH, boxes_subslice)
                .expect("Could not construct bounding box handler");
        bounding_boxes.change_own_bounding_type(BoundingBoxType::XYXY);
        let box_indices = bounding_boxes
            .tensor
            .nms(score_tensor.clone().squeeze(), nms_options);
        let remaining_boxes = box_tensor.select(1, box_indices.clone());
        debug!(
            ?box_indices,
            ?remaining_boxes,
            "Finished Non-Maxium Suppression"
        );
        let out: Vec<f32> = remaining_boxes
            .iter_dim(1)
            .map(|el| el.squeeze_dims::<1>(&[0, 1]))
            .next()
            .expect("Expected to get at least one box")
            .into_data()
            .to_vec()
            .unwrap();
        assert!(out.len() >= 4);
        let x_center = out[0];
        let y_center = out[1];
        let w = out[2];
        let h = out[3];

        let with_drawn_box = imageproc::drawing::draw_hollow_rect(
            &resized,
            imageproc::rect::Rect::at((x_center - w / 2.0) as i32, (y_center - h / 2.0) as i32)
                .of_size(w as u32, h as u32),
            image::Rgba::<u8>([127, 0, 127, 0]),
        );
        with_drawn_box
            .save("test_with_box.png")
            .expect("Could not save image.");
        debug!("Successfully saved image");

        info!("Test succeeded without issues");
    }
}
