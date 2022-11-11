import cv2
import numpy as np
from pathlib import Path

# 세점 사이의 각도 계산
def get_angle(p1: list, p2: list, p3: list, angle_vec: bool) -> float:
    rad = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - np.arctan2(
        p1[1] - p2[1], p1[0] - p2[0]
    )
    deg = rad * (180 / np.pi)
    if angle_vec:
        deg = 360 - abs(deg)
    return abs(deg)


# BODY25에서 각 파트 번호, 선으로 연결될 POSE_PAIRS
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
protoFile = "C:\\Users\\gram\\openpose\\models\\pose\\body_25\\pose_deploy.prototxt"
weightsFile = (
    "C:\\Users\\gram\\openpose\\models\\pose\\body_25\\pose_iter_584000.caffemodel"
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
count_foot = 0
r_shoulder = 0
l_shoulder = 0
r_hip = 0
l_hip = 0
total_r_angle = 0
total_l_angle = 0
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
    x_data, y_data = [], []
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
                x_data.append(x)
                y_data.append(y)

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
                x_data.append(0)
                y_data.append(0)

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
                x_data.append(x)
                y_data.append(y)

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
                x_data.append(0)
                y_data.append(0)

    # 오른쪽 발목 각도 계산
    r1 = [x_data[10], y_data[10]]
    r2 = [x_data[11], y_data[11]]
    r3 = [x_data[22], y_data[22]]
    angle_r = get_angle(r1, r2, r3, True)
    total_r_angle += angle_r

    # 왼쪽 발목 각도 계산
    l1 = [x_data[13], y_data[13]]
    l2 = [x_data[14], y_data[14]]
    l3 = [x_data[19], y_data[19]]
    angle_l = get_angle(l1, l2, l3, False)
    total_l_angle += angle_l

    # 좌우 어깨 좌표
    l_shoulder += y_data[5]
    r_shoulder += y_data[2]

    # 좌우 골반 좌표
    l_hip += y_data[12]
    r_hip += y_data[9]

    print(
        round(angle_l, 2),
        " ",
        round(angle_r, 2),
        " ",
        round(l_shoulder, 2),
        " ",
        round(r_shoulder, 2),
        " ",
        round(l_hip, 2),
        " ",
        round(r_hip, 2),
    )
    if angle_r <= 165 and angle_l <= 165:
        count_foot += 1

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

# 걸음걸이 구분
# print("평균 왼쪽 발목 각도: ", abs(round(180 - total_l_angle / total_frame, 2)))
# print("평균 오른쪽 발목 각도: ", abs(round(180 - total_r_angle / total_frame, 2)))
# if count_foot >= int(total_frame * 0.3):
#     print("팔자 걸음")
# else:
#     print("일자 걸음")

# 어깨 대칭 구분
# if abs(l_shoulder / total_frame - r_shoulder / total_frame) >= 1:
#     if l_shoulder / total_frame - r_shoulder / total_frame > 0:
#         print("왼쪽 어깨 올라감")
#     else:
#         print("오른쪽 어깨 올라감")
# else:
#     print("어깨의 균형이 잘 맞음")

# 골반 대칭 구분
# if abs(l_hip / total_frame - r_hip / total_frame) >= 1:
#     if l_hip / total_frame - r_hip / total_frame > 0:
#         print("왼쪽 골반이 올라감")
#     else:
#         print("오른쪽 골반이 올라감")
# else:
#     print("골반 균형이 잘 맞음")

# print(total_frame)
capture.release()  # 카메라 장치에서 받아온 메모리 해제
cv2.destroyAllWindows()  # 모든 윈도우 창 닫음
