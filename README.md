영상 내 과적차량 감지 모델
=========================
프로젝트 개요
------------

- 프로젝트 배경 : 영상에서 과적차량을 인식하여 단속하는 서비스를 개발하고자 한다
- 이용 데이터 : 과적 차량(500장)과 정상 차량(500장)의 이미지(jpg) 및 라벨링 데이터(json)
- 사용 모델 : yolov5s6

패키지 정보
-------------
해당 패키지는 사전 학습한 모델을 사용해 영상 내 다양한 객체를 탐지하고 결과 영상을 만들어내는 기능이 구현되어 있습니다.

패키지의 디렉토리 구조는 다음과 같습니다.
```
.
├── data
│   ├── A01_B02_C00_D01_0703_E08_F03_554_1.jpg
│   ├── A01_B02_C00_D01_0703_E08_F03_568_3.jpg
│   ├── sample.mp4
│   └── sample2.mp4
├── models
│   └── best.onnx
├── results
│   ├── img_554.jpg
│   ├── img_568.jpg
│   ├── output.mp4
│   └── output2.mp4
├── test
│   └── test.ipynb
├── utils
│   ├── __init__.py
│   └── detection.py
├── AI_16기_한승희_CP1_DS.ipynb
├── README.md
├── detect.py
└── requirements.txt
```

### data
모델을 test 할 수 있는 sample data 입니다.
### models
onnx 형태의 사전학습된 model (`best.onnx`) 이 저장되어 있습니다.</br>
[학습한 데이터 파일](https://drive.google.com/file/d/11TI52Dwm135_VKmnOEQa9WbZYyWCBHHZ/view?usp=share_link)</br>
모델 학습과정은 `AI_16기_한승희_CP1_DS.ipynb` 에서 볼 수 있습니다.
### results
`detection.py`에 의해 생성된 결과 영상/이미지가 해당 폴더에 저장됩니다.</br>
영상은 .mp4, 이미지는 jpg 형태로 저장됩니다.
### test
해당 패키지가 정상적으로 작동하는지 확인할 수 있는 `test.ipynb` 파일을 포함하고 있습니다.
### utils
`best.onnx` 모델을 통해 객체탐지를 수행하고 결과를 저장하는 `detection.py` 가 저장되어 있습니다.


패키지 사용하기
=====
### 사용 환경
> - Python 3.8.16</br>
> - conda 22.9.0

**터미널**에서 다음을 실행해주세요.
```
git clone https://github.com/H-Seung/AI_16_HanSeungHee_CP1_DS.git
cd AI_16_HanSeungHee_CP1_DS
pip install -r requirements.txt
```
파일 `detect.py`에서 detect를 수행할 데이터의 경로(input_path)와 결과가 저장될 파일 경로(output_path)를 설정해주세요.</br>
아래는 `detect.py` 의 코드입니다.</br>(해당 코드에서 input_path,output_path는 테스트할 수 있도록 설정되어 있습니다. 원하는 경로로 수정해주세요.)
```
from utils.detection import Detection

model_path = './models/best.onnx'
d = Detection(model_path)

input_path = './data/sample2.mp4'
output_path = './results/output2.mp4'
d.detect(input_path, output_path)
```
**터미널**에서 다음을 실행해주세요.
```
python detect.py
```
객체탐지를 수행한 결과 이미지 또는 영상이 output_path에 저장됩니다.

※ 동일한 수행을 `test/test.ipynb`에서 Interactive 형태로 작동을 확인할 수 있습니다.