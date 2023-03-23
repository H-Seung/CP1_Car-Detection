from utils.detection import Detection
from pathlib import Path
import argparse
import sys
import os

FILE = Path(__file__).resolve() # 현 file의 경로
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def run(
    model_path = ROOT / 'models/best.onnx',
    input = ROOT / 'data',
    output = ROOT/ 'results',
    conf = 0.4
):
    model_path = './models/best.onnx'
    d = Detection(model_path, conf)

    # 해당 경로에 있는 모든 파일에 대해 detect 수행
    for file in os.listdir(input):
        input_path = os.path.join(input, file)
        output_path = os.path.join(output, file)
        d.detect(input_path, output_path)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=ROOT / 'models/best.onnx')
    parser.add_argument('--input', type=str, default= ROOT / 'data') # 객체탐지를 진행할 폴더 경로
    parser.add_argument('--output', type=str, default= ROOT/ 'results') # 객체탐지 결과가 저장될 폴더 경로
    parser.add_argument('--conf', type=float, default= 0.4)
    args = parser.parse_args()
    return args

def main(args):
    run(**vars(args))
    
if __name__ == '__main__':
    args = parse_opt()
    main(args)

# python detect.py 실행할 경우 - default값으로 정해둔 data 폴더 내 모든 파일에 대해 detect 수행
# python detection.py --onnx_path [가중치 파일 경로] --source [데이터 파일 경로] --output [결과 저장파일 경로] --conf [confidence threshold]
# ex) python detect.py --input /c/Users/LG/Desktop/test11 --output /c/Users/LG/Desktop/test11