영상 내 과적차량 감지 모델
=========================
프로젝트 개요
------------

- 프로젝트 배경 : 영상에서 과적차량을 인식하여 단속하는 서비스를 개발하고자 한다
- 이용 데이터 : 과적 차량(500장)과 정상 차량(500장)의 이미지(jpg) 및 라벨링 데이터(json)
- 사용 모델 : yolov5s6
- 학습에 이용한 데이터 : [AI HUB - 과적차량 도로 위험 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=530)
  ![image](https://github.com/H-Seung/CP1_Car-Detection/assets/114974542/57fe6c49-01df-4067-af68-9800060df7d1)

모델 학습 및 성능 평가
------------
- 모델 학습
  ```
  !python train.py --img 832 --batch 32 --epochs 50 --data ./vehicle/dataset.yaml\
  --cfg ./models/custom_yolov5s6.yaml\ --weights ''\
  ```
- 모델 성능
  ![image](https://github.com/H-Seung/CP1_Car-Detection/assets/114974542/ef4ec403-a0bc-4416-ba8d-172bc0193239)


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
detect를 수행할 데이터들의 폴더 경로(input)와 결과가 저장될 폴더 경로(output), 객체 탐지 임계값(conf) 인자를 설정하여 실행할 수 있습니다.</br>
python detection.py --model_path [가중치 파일 경로] --input [데이터 폴더 경로] --output [결과 저장폴더 경로] --conf [confidence threshold]

```
# 예시
python detect.py --model_path ./models/best.onnx
                 --input /c/Users/LG/Desktop/test1
                 --output /c/Users/LG/Desktop/test2
                 --conf 0.3
```
또는 인자값을 설정하지 않고 default 값으로 실행할 수 있습니다.
```
# 예시
python detect.py
```
설정된 폴더 경로(input) 내의 모든 이미지 또는 영상파일(.jpg, .mp4)에 대해 model_path의 가중치 파일(.onnx)을 이용하여, 객체탐지를 수행한 결과 이미지 또는 영상이 output 폴더에 저장됩니다.

※ 동일한 수행을 `test/test.ipynb`에서 Interactive 형태로 작동을 확인할 수 있습니다.
