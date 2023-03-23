import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from tqdm import tqdm

class Detection:
    def __init__(self, model_path, conf):
        self.model_path = model_path
        self.INPUT_WIDTH = 832  # input image의 width ※ onnx 로 내보낼때 832로 설정했음
        self.INPUT_HEIGHT = 832  # input image의 height
        self.CONF_THRESHOLD = conf  # 객체 필터링 임계값
        self.SCORE_THRESHOLD = 0.5  # class 필터링 임계값
        self.NMS_THRESHOLD = 0.45  # 겹치는 box 제거 임계값
        self.labels = {0:'적재불량',1:'정상차량'}
        

    def process(self, frame):
        # YOLO 네트워크 불러오기
        YOLO_net = cv2.dnn.readNet(self.model_path)
        ## 전체 layer에서 detect된 Output layer만 filtering
        layer_names = YOLO_net.getLayerNames()  # 335 개
        outlayer_names = [layer_names[i - 1] for i in YOLO_net.getUnconnectedOutLayers()]  # >'output0' 레이어 하나
        
        # YOLO에 frame 입력(입력영상을 네트워크에 넣기위해 blob으로 만듦. size: 입력이미지 resize 크기)
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(self.INPUT_WIDTH, self.INPUT_HEIGHT), swapRB=True, crop=False)
        YOLO_net.setInput(blob)  # 네트워크 입력 설정

        # 순전파(추론) 실행
        Outputs = YOLO_net.forward(outlayer_names)  # -> (array([[[xc, yc, w, h, confidence, probability_distribution(class 1,2...)],...]])) confidence: 박스가 객체를 포함하고 있을 확률, proba~: 객체가 각 클래스일 확률

        # 객체탐지 변수 초기화
        class_ids = []
        confidences = []
        boxes = []

        # 영상의 frame 크기 가져오기
        img_H, img_W = frame.shape[:2]   # 테스트 샘플영상 사이즈 (720, 480)

        for output in Outputs[0]:   # per output layer (여기선 한번)
            for detection in output:  # per Detected Object
                # bad detection은 버림
                if detection[4] >= self.CONF_THRESHOLD: # 객체일 확률
                    class_scores = detection[5:]  # class별 확률
                    class_id = np.argmax(class_scores)  # 확률 높은 값의 인덱스를 class_id로 저장
                    class_conf = class_scores[class_id]  # 해당 클래스일 확률
                    # SCORE_THRESHOLD보다 높은 class 확률이면 수행
                    if class_conf > self.SCORE_THRESHOLD: 
                        # detection: x중심, y중심, 너비, 높이(INPUT SIZE 기준)-> 원본 이미지에 맞게 scaling
                        xc = int(detection[0] * img_W/self.INPUT_WIDTH)
                        yc = int(detection[1] * img_H/self.INPUT_HEIGHT)
                        width = int(detection[2] * img_W/self.INPUT_WIDTH)
                        height = int(detection[3] * img_H/self.INPUT_HEIGHT)
                        # 원본 이미지 좌표와 동일하게, 좌상단 x,y 좌표 생성 (※원점:좌상단)
                        left = int(xc - width/2)
                        top = int(yc - height/2)

                        # output layer별로 Detected Object에 대한 class id, class_conf, 좌표 저장
                        class_ids.append(class_id)
                        confidences.append(float(class_conf))
                        boxes.append([left, top, width, height])
        
        # 비최대억제(NMS) 알고리즘을 적용해 확률이 가장 높은 박스만 남김
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.SCORE_THRESHOLD, self.NMS_THRESHOLD)
                
        # bbox/Text 설정값
        BLUE=(255, 0, 0)
        RED=(0, 0, 255)
        WHITE=(255,255,255)
        # colors = np.random.uniform(0, 255, size=(len(labels), 3))
        colors = [RED, BLUE]  # 불법:red, 정상:blue

        # 이미지에 box, label 그리기
        for i in range(len(boxes)): # per detection
            if i in idxs:   # nms로 filtering된 box index
                color = colors[class_ids[i]]  # colors[클래스번호]
                # Class label
                label = f"{self.labels[class_ids[i]]} {confidences[i]:.3f}"
                # Bounding box
                box = boxes[i]
                left, top, width, height = box[0],box[1],box[2],box[3]
                start_point = (int(left), int(top))
                end_point = (int(left+width), int(top+height))
                cv2.rectangle(frame, start_point, end_point, color=color, thickness=2, lineType=cv2.LINE_AA) # bbox
                # text 및 text 배경
                if not is_ascii(label): # 한글 label
                    frame = Image.fromarray(frame)
                    draw = ImageDraw.Draw(frame)
                    FONT = ImageFont.truetype('Fonts/gulim.ttc', 
                                              max(round(sum(frame.size) / 2 * 0.03), 12)) # 12와 h,w 합 중 max
                    if label:
                        # tw, th = FONT.getsize(label)  # text width, height (WARNING: deprecated) in PIL 9.2.0
                        _, _, tw, th = FONT.getbbox(label)  # text width, height (New)
                        outside = top - th >= 0  # label fits outside box
                        # Text 배경 채우기
                        draw.rectangle(
                            (left-1, top-th-1 if outside else top, left+tw+1,
                            top+1 if outside else top+th+1),
                            fill=color
                        )
                        # Text 넣기
                        # draw.text((left, top), label, fill=txt_color, font=FONT, anchor='ls')  # for PIL>8.0
                        draw.text((left-1, top-th-1 if outside else top), label, fill=WHITE, font=FONT)
                    frame = np.array(frame)  # OpenCV가 처리가능하도록 다시 np array로 변환

                else: # cv2
                    FONT_FACE= cv2.FONT_HERSHEY_SIMPLEX  # 폰트 종류
                    FONT_SCALE = 0.7  # 폰트 크기   
                    # cv2.putText(frame, label, (int(left), int(top - 5)), 0, 0.5, (0,0,0), 1)
                    if label:
                        (tw, th), baseline = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, thickness=1) # text size
                        print(tw, th, baseline)
                        # Text 배경 채우기
                        outside = top - th >= 3
                        p1 = left, top
                        p2 = left + tw, top - th - 3 if outside else top + th + 3
                        cv2.rectangle(frame, p1, p2, color, -1, cv2.LINE_AA)  # filled
                        # Text 넣기
                        cv2.putText(frame, label, (left, top-2 if outside else top+th+2), 
                                    FONT_FACE, FONT_SCALE, WHITE, thickness=1, lineType=cv2.LINE_AA)

        return frame
    

    def detect(self, input_path, output_path):   
        # 영상 신호 받기
        VideoSignal = cv2.VideoCapture(input_path)

        # input(jpg/mp4)에 따른 output 설정
        if input_path.endswith(".mp4"):
            codec =cv2.VideoWriter_fourcc(*"mp4v")
            vid_fps = VideoSignal.get(cv2.CAP_PROP_FPS)
            vid_size = (int(VideoSignal.get(cv2.CAP_PROP_FRAME_WIDTH)), int(VideoSignal.get(cv2.CAP_PROP_FRAME_HEIGHT)))

            vid_writer = cv2.VideoWriter(output_path, codec, vid_fps, vid_size) 
            
            frame_cnt = int(VideoSignal.get(cv2.CAP_PROP_FRAME_COUNT))
            print('총 Frame 수:', frame_cnt)

            # 객체 탐지 및 결과 저장
            with tqdm(total=frame_cnt) as pbar:
                while True:
                    pbar.update(1)
                    ret, frame = VideoSignal.read()  # 객체 정보 읽음. ret:bool, frame:array(h w c)
                    if not ret:
                        break
                    frame = self.process(frame.copy())
                    vid_writer.write(frame) # 결과 저장
            vid_writer.release() # release() : open한 캡쳐 객체를 해제
            VideoSignal.release()

            # # 결과 영상 출력 (마지막frame)
            # VideoSignal = cv2.VideoCapture(output_path)
            # while VideoSignal.isOpened():
            #     ret, vid = VideoSignal.read()
            #     if not ret:
            #         break
            #     print(vid)
            #     cv2.imshow('output', vid)
            #     if cv2.waitKey(33) > 0: break # 0: key 입력이 있을 때까지 대기 
            # VideoSignal.release()
            # cv2.destroyAllWindows() # window 닫기

        elif input_path.endswith(".jpg"):
            frame = cv2.imread(input_path) # 이미지 읽어오기
            img = self.process(frame)
            # 결과 이미지 저장
            cv2.imwrite(output_path, img)
            # 결과 이미지를 window에 출력
            cv2.namedWindow('output', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO) # window 크기 조절 허용
            cv2.imshow('output', img)
            cv2.waitKey(0) # 0: key 입력이 있을 때까지 대기 / X 버튼으로 나오기
            cv2.destroyAllWindows() # window 닫기
        
        else:
            print("input_path 확장자 확인 필요")


def is_ascii(s=''):
    # Is string composed of all ASCII (no UTF) characters? (note str().isascii() introduced in python 3.7)
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode('ascii', 'ignore')) == len(s)
