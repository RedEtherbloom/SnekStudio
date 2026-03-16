use burn_onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("unprocessed_models/face_blendshapes.onnx")
        .input("unprocessed_models/face_detector.onnx")
        .input("unprocessed_models/face_landmarks_detector.onnx")
        .input("unprocessed_models/hand_detector.onnx")
        .input("unprocessed_models/hand_landmarks_detector.onnx")
        .input("unprocessed_models/pose_landmarks_detector.onnx")
        .out_dir("models/")
        .run_from_script();
}
