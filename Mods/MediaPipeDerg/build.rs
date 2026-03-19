use burn_onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("onnx_models/face_detector.onnx")
        .input("onnx_models/face_landmarks_detector.onnx")
        .input("onnx_models/face_blendshapes.onnx")
        .input("onnx_models/hand_detector.onnx")
        .input("onnx_models/hand_landmarks_detector.onnx")
        .out_dir("models/")
        .run_from_script();
}
