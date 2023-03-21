from utils.detection import Detection

model_path = './models/best.onnx'
d = Detection(model_path)

input_path = './data/sample2.mp4'
output_path = './results/output2.mp4'
d.detect(input_path, output_path)