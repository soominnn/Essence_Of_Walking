import cv2
import numpy as np
import pandas as pd
from pathlib import Path


def get_angle(p1: list, p2: list, p3: list, angle_vec: bool) -> float:
    rad = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - np.arctan2(
        p1[1] - p2[1], p1[0] - p2[0]
    )
    deg = rad * (180 / np.pi)
    if angle_vec:
        deg = 360 - abs(deg)
    return abs(deg)


def output_keypoints_with_lines_video(
    proto_file, weights_file, video_path, threshold, BODY_PARTS, POSE_PAIRS
):

    # 네트워크 불러오기
    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

    # GPU 사용
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # 비디오 읽어오기
    capture = cv2.VideoCapture(video_path)
    # total_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))

    ret, frame = capture.read()
    frame_height = frame.shape[1]
    frame_width = frame.shape[0]
    # (frame_height, frame_width) = frame.shape[:2]
    h = 700
    w = int(((h / 2) / frame_height) * frame_width) * 2

    image_height = 368
    image_width = 368

    out_path = "output.mp4"
    out = cv2.VideoWriter(out_path, 0, fps, (w, h))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    (f_h, f_w) = (h, w)
    zeros = None

    # 분석을 위한 변수 선언
    # total_frame = 0
    # count_foot = 0
    r_shoulder = 0
    l_shoulder = 0
    r_hip = 0
    l_hip = 0
    total_r_angle = 0
    total_l_angle = 0

    while True:
        # now_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = capture.read()
        if ret != True:
            break

        frame = cv2.resize(frame, (w, h), cv2.INTER_AREA)

        # 네트워크에 넣기 위한 전처리
        input_blob = cv2.dnn.blobFromImage(
            frame,
            1.0 / 255,
            (image_width, image_height),
            (0, 0, 0),
            swapRB=False,
            crop=False,
        )

        # 전처리된 blob 네트워크에 입력
        net.setInput(input_blob)

        # 결과 받아오기
        out = net.forward()
        out_height = out.shape[2]
        out_width = out.shape[3]

        # 원본 이미지의 높이, 너비를 받아오기
        # frame_height, frame_width = frame.shape[:2]

        # 키포인트를 저장할 빈 리스트 및 초기화
        points = []
        x_data, y_data = [], []

        for i in range(25):
            # 신체 부위의 confidence map
            prob_map = out[0, i, :, :]

            # 최소값, 최대값, 최소값 위치, 최대값 위치
            min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

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

            # 원본 이미지에 맞게 포인트 위치 조정
            x = (w * point[0]) / out_width
            x = int(x)
            y = (h * point[1]) / out_height
            y = int(y)
            if BODY_PARTS[i] not in without:
                if prob > threshold:  # [pointed]
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

                else:  # [not pointed]
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

            else:
                if prob > threshold:  # [pointed]
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

                else:  # [not pointed]
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
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h), True)
            zeros = np.zeros((h, w), dtype="uint8")

        writer.write(cv2.resize(frame, (w, h)))
        cv2.imshow("Output_Keypoints", frame)

        if cv2.waitKey(10) == 27:  # esc 입력시 종료
            break

    capture.release()  # 카메라 장치에서 받아온 메모리 해제
    cv2.destroyAllWindows()  # 모든 윈도우 창 닫음


BODY_PARTS_BODY_25 = {
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

POSE_PAIRS_BODY_25 = [
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
# 신경 네트워크의 구조를 지정하는 prototxt 파일 (다양한 계층이 배열되는 방법 등)
protoFile_body_25 = (
    "C:\\Users\\gram\\openpose\\models\\pose\\body_25\\pose_deploy.prototxt"
)

# 훈련된 모델의 weight 를 저장하는 caffemodel 파일
weightsFile_body_25 = (
    "C:\\Users\\gram\\openpose\\models\\pose\\body_25\\pose_iter_584000.caffemodel"
)

# 비디오 경로
video = "k.mp4"

output_keypoints_with_lines_video(
    proto_file=protoFile_body_25,
    weights_file=weightsFile_body_25,
    video_path=video,
    threshold=0.1,
    BODY_PARTS=BODY_PARTS_BODY_25,
    POSE_PAIRS=POSE_PAIRS_BODY_25,
)
