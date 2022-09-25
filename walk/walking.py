import cv2
import numpy as np
from pathlib import Path

# MPII에서 각 파트 번호, 선으로 연결될 POSE_PAIRS
BODY_PARTS = {
    0: "Nose",
    1: "Neck",
    2: "RShoulder",
    3: "RElbow",
    4: "RWrist",
    5: "LShoulder",
    6: "LElbow",
    7: "LWrist",
    8: "MidHip",
    9: "RHip",
    10: "RKnee",
    11: "RAnkle",
    12: "LHip",
    13: "LKnee",
    14: "LAnkle",
    15: "REye",
    16: "LEye",
    17: "REar",
    18: "LEar",
    19: "LBigToe",
    20: "LSmallToe",
    21: "LHeel",
    22: "RBigToe",
    23: "RSmallToe",
    24: "RHeel",
    25: "Background",
}

POSE_PAIRS = [
    [0, 1],
    [1, 2],
    [1, 5],
    [1, 8],
    [8, 9],
    [8, 12],
    [9, 10],
    [12, 13],
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 7],
    [10, 11],
    [11, 22],
    [13, 14],
    [14, 19],
]

# 각 파일 path
BASE_DIR = Path(__file__).resolve().parent
protoFile = "./pose_deploy.prototxt"
weightsFile = (
    "./pose_iter_584000.caffemodel"
)
# 위의 path에 있는 network 모델 불러오기
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# 쿠다 사용 안하면 밑에 이미지 크기를 줄이는게 나을 것이다
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) #벡엔드로 쿠다를 사용하여 속도향상을 꾀한다
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA) # 쿠다 디바이스에 계산 요청


###카메라랑 연결...?
capture = cv2.VideoCapture(0)  # 카메라 정보 받아옴
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640) #카메라 속성 설정
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # width:너비, height: 높이

# 동영상의 프레임 속도
fps = int(capture.get(cv2.CAP_PROP_FPS)) / 10

inputWidth = 368
inputHeight = 368
inputScale = 1.0 / 255

# h = 800
# w = int(((h / 2) / frameHeight) * frameWidth) * 2

# 비디오 저장 위치
out_path = "output.mkv"
out = cv2.VideoWriter(out_path, 0, fps, (368, 368))

# 인코딩 방식 설정
fourcc = cv2.VideoWriter_fourcc(*"MPEG")
writer = None
zeros = None

total_frame = 0
# 반복문을 통해 카메라에서 프레임을 지속적으로 받아옴
while cv2.waitKey(1) < 0:  # 아무 키나 누르면 끝난다.
    # 웹캠으로부터 영상 가져옴
    total_frame += 1
    hasFrame, frame = capture.read()

    # 영상이 커서 느리면 사이즈를 줄이자
    # frame=cv2.resize(frame,dsize=(320,240),interpolation=cv2.INTER_AREA)

    # 웹캠으로부터 영상을 가져올 수 없으면 웹캠 중지
    if not hasFrame:
        cv2.waitKey()
        break

    #
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(
        frame,
        inputScale,
        (inputWidth, inputHeight),
        (0, 0, 0),
        swapRB=False,
        crop=False,
    )

    imgb = cv2.dnn.imagesFromBlob(inpBlob)
    # cv2.imshow("motion",(imgb[0]*255.0).astype(np.uint8))

    # network에 넣어주기
    net.setInput(inpBlob)

    # 결과 받아오기
    output = net.forward()

    # 키포인트 검출시 이미지에 그려줌
    points = []
    for i in range(0, 25):
        # 해당 신체부위 신뢰도 얻음.
        probMap = output[0, i, :, :]

        # global 최대값 찾기
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # 표시하지 않을 key points
        without = [
            "REye",
            "LEye",
            "REar",
            "LEar",
            "LSmallToe",
            "LHeel",
            "RHeel",
            "RSmallToe",
            "Background",
        ]

        # 원래 이미지에 맞게 점 위치 변경
        x = (frameWidth * point[0]) / output.shape[3]
        x = int(x)
        y = (frameHeight * point[1]) / output.shape[2]
        y = int(y)

        # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로
        if BODY_PARTS[i] not in without:
            if prob > 0.1:  # keypoint 검출
                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 255),
                    thickness=-1,
                    lineType=cv2.FILLED,
                )
                cv2.putText(
                    frame,
                    str(i),
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    1,
                    lineType=cv2.LINE_AA,
                )

                points.append((x, y))

            else:  # key point가 검출이 되지 않았을 때
                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 255),
                    thickness=-1,
                    lineType=cv2.FILLED,
                )
                cv2.putText(
                    frame,
                    str(i),
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    1,
                    lineType=cv2.LINE_AA,
                )

                points.append(None)

        # keypoint가 without에 해당되는 경우 영상에 표시하지 않음
        else:
            if prob > 0.1:  # keypoint 검출
                cv2.circle(
                    frame,
                    (x, y),
                    0,
                    (0, 255, 255),
                    thickness=-1,
                    lineType=cv2.FILLED,
                )
                points.append((x, y))

            else:  # key point가 검출이 되지 않았을 때
                cv2.circle(
                    frame,
                    (x, y),
                    0,
                    (0, 255, 255),
                    thickness=-1,
                    lineType=cv2.FILLED,
                )
                points.append(None)

    # 각 POSE_PAIRS별로 선 그어줌 (머리 - 목, 목 - 왼쪽어깨, ...)
    # 각 pair를 line으로 연결
    for pair in POSE_PAIRS:
        part_a = pair[0]
        part_b = pair[1]

        foot = [10, 11, 13, 14]
        hip = [[8, 9], [8, 12]]
        shoulder = [[1, 2], [1, 5]]

        if pair[0] in foot:
            if points[part_a] and points[part_b]:
                cv2.line(frame, points[part_a], points[part_b], (0, 0, 255), 4)

        elif pair in hip:
            if points[part_a] and points[part_b]:
                cv2.line(frame, points[part_a], points[part_b], (255, 0, 75), 4)

        elif pair in shoulder:
            if points[part_a] and points[part_b]:
                cv2.line(frame, points[part_a], points[part_b], (255, 0, 0), 4)

        else:
            if points[part_a] and points[part_b]:
                cv2.line(frame, points[part_a], points[part_b], (0, 255, 0), 4)

    if writer is None:
        writer = cv2.VideoWriter(out_path, fourcc, fps, (368, 368), True)
        zeros = np.zeros((368, 368), dtype="uint8")

    writer.write(cv2.resize(frame, (368, 368)))
    cv2.imshow("Output-Keypoints", frame)

print(total_frame)
capture.release()  # 카메라 장치에서 받아온 메모리 해제
cv2.destroyAllWindows()  # 모든 윈도우 창 닫음