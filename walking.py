import cv2
import numpy as np
import pandas as pd
import progressbar

def get_angle(p1: list, p2: list, p3: list, angle_vec: bool) -> float:
    rad = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - np.arctan2(
        p1[1] - p2[1], p1[0] - p2[0]
    )
    deg = rad * (180 / np.pi)
    if angle_vec:
        deg = 360 - abs(deg)
    return abs(deg)


def output_keypoints_with_lines_video(proto_file, weights_file, video_path, threshold, BODY_PARTS, POSE_PAIRS):
    # 네트워크 불러오기
    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

    # GPU 사용
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # 비디오 읽어오기
    capture = cv2.VideoCapture(video_path)
    total_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # now_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
    fps = int(capture.get(cv2.CAP_PROP_FPS))

    ret, frame = capture.read()

    # 원본 이미지의 높이, 너비를 받아오기
    (frame_height, frame_width) = frame.shape[:2]
    h = 800
    w = int(((h / 2) / frame_height) * frame_width) * 2
    image_height = 368
    image_width = 368
    out_path = "output.mkv"
    out = cv2.VideoWriter(out_path, 0, fps, (w, h))

    fourcc = cv2.VideoWriter_fourcc(*"MPEG")
    writer = None
    # (f_h, f_w) = (h, w)
    zeros = None

    # CSV 저장할 빈 리스트
    data = []
    previous_x, previous_y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Set up the progressbar
    widgets = ["--[INFO]-- Analyzing Video: ", progressbar.Percentage(), " ", progressbar.Bar(), " ",
               progressbar.ETA(), ]
    pbar = progressbar.ProgressBar(maxval=total_frame, widgets=widgets).start()
    p = 0
    count_foot = 0
    r_shoulder = 0
    l_shoulder = 0
    r_hip = 0
    l_hip = 0
    while True:
        now_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = capture.read()

        if ret != True:
            break

        frame = cv2.resize(frame, (w, h), cv2.INTER_AREA)
        # 네트워크에 넣기 위한 전처리
        input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (image_width, image_height), (0, 0, 0),
                                           swapRB=False, crop=False, )

        # 전처리된 blob 네트워크에 입력
        net.setInput(input_blob)

        # 결과 받아오기
        out = net.forward()
        out_height = out.shape[2]
        out_width = out.shape[3]

        # 키포인트를 저장할 빈 리스트 및 초기화
        points = []
        x_data, y_data = [], []

        print(f"============================== frame: {now_frame:.0f} / {total_frame:.0f} ==============================")

        for i in range(25):
            # print(np.shape(out))
            # 신체 부위의 confidence map
            prob_map = out[0, i, :, :]

            # 최소값, 최대값, 최소값 위치, 최대값 위치
            min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

            without = ["REye", "LEye", "REar", "LEar", "LSmallToe", "LHeel", "RHeel", "RSmallToe"]

            # 원본 이미지에 맞게 포인트 위치 조정
            x = (w * point[0]) / out_width
            x = int(x)
            y = (h * point[1]) / out_height
            y = int(y)
            if BODY_PARTS[i] not in without:
                if prob > threshold:  # [pointed]
                    cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED, )
                    cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                                1, lineType=cv2.LINE_AA, )

                    points.append((x, y))
                    # print(f"[pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")
                    x_data.append(x)
                    y_data.append(y)

                else:  # [not pointed]
                    cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED, )
                    cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1,
                                lineType=cv2.LINE_AA, )

                    points.append(None)
                    # print(f"[not pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")
                    x_data.append(previous_x[i])
                    y_data.append(previous_y[i])

            else:
                if prob > threshold:  # [pointed]
                    cv2.circle(frame, (x, y), 0, (0, 255, 255), thickness=-1, lineType=cv2.FILLED, )
                    points.append((x, y))
                    # print(f"[pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")
                    x_data.append(x)
                    y_data.append(y)

                else:  # [not pointed]
                    cv2.circle(frame, (x, y), 0, (0, 255, 255), thickness=-1, lineType=cv2.FILLED, )
                    points.append(None)
                    # print(f"[not pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")
                    x_data.append(previous_x[i])
                    y_data.append(previous_y[i])
        # 각도 계산
        r1 = [x_data[10], y_data[10]]
        r2 = [x_data[11], y_data[11]]
        r3 = [x_data[22], y_data[22]]
        print('type',type(r1))
        angle_r = get_angle(r1, r2, r3, True)

        l1 = [x_data[13], y_data[13]]
        l2 = [x_data[14], y_data[14]]
        l3 = [x_data[19], y_data[19]]
        angle_l = get_angle(l1, l2, l3, False)

        l_shoulder += y_data[5]
        r_shoulder += y_data[2]

        l_hip += y_data[12]
        r_hip += y_data[9]

        if angle_r <= 165 and angle_l <= 165:
            count_foot += 1
            print(count_foot)
        print(angle_r, angle_l)

        for pair in POSE_PAIRS:
            part_a = pair[0]  # 0 (Head)
            part_b = pair[1]  # 1 (Neck)

            foot = [10, 11, 13, 14]
            hip = [[8, 9], [8, 12]]
            shoulder = [[1, 2], [1, 5]]

            if pair[0] in foot:
                if points[part_a] and points[part_b]:
                    # print(f"[linked] {part_a} {points[part_a]} <=> {part_b} {points[part_b]}")
                    cv2.line(frame, points[part_a], points[part_b], (0, 0, 255), 4)
                # else:
                #     print(f"[not linked] {part_a} {points[part_a]} <=> {part_b} {points[part_b]}")

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

        data.append(x_data + y_data)
        previous_x, previous_y = x_data, y_data

        p += 1
        pbar.update(p)

        if cv2.waitKey(10) == 27:  # esc 입력시 종료
            break

    # 발 각도
    if count_foot >= int(total_frame * 0.3):
        print("팔자 걸음")

    # 어깨 각도
    if abs(l_shoulder / total_frame - r_shoulder / total_frame) >= 1:
        if l_shoulder / total_frame - r_shoulder / total_frame > 0:
            print("왼쪽 어깨 올라감")
        else:
            print("오른쪽 어깨 올라감")
    else:
        print("어깨의 균형이 잘 맞음")

    if abs(l_hip / total_frame - r_hip / total_frame) >= 1:
        if l_hip / total_frame - r_hip / total_frame > 0:
            print("왼쪽 엉덩이가 올라감")
        else:
            print("오른쪽 엉덩이가 올라감")
    else:
        print("골반 균형이 잘 맞음")
    csv_path = "output.csv"
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print("save complete")

    pbar.finish()
    capture.release()
    cv2.destroyAllWindows()


BODY_PARTS_BODY_25 = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist", 5: "LShoulder",
                      6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip", 10: "RKnee", 11: "RAnkle", 12: "LHip",
                      13: "LKnee",
                      14: "LAnkle", 15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe", 20: "LSmallToe",
                      21: "LHeel",
                      22: "RBigToe", 23: "RSmallToe", 24: "RHeel", 25: "Background", }

POSE_PAIRS_BODY_25 = [[0, 1], [1, 2], [1, 5], [1, 8], [8, 9], [8, 12], [9, 10], [12, 13], [2, 3],
                      [3, 4], [5, 6], [6, 7], [10, 11], [11, 22], [13, 14], [14, 19]]

# 신경 네트워크의 구조를 지정하는 prototxt 파일 (다양한 계층이 배열되는 방법 등)
protoFile_body_25 = "model/pose/body_25/pose_deploy.prototxt"

# 훈련된 모델의 weight 를 저장하는 caffemodel 파일
weightsFile_body_25 = "model/pose/body_25/pose_iter_584000.caffemodel"

# 비디오 경로
video = "test/50.mp4"

output_keypoints_with_lines_video(proto_file=protoFile_body_25, weights_file=weightsFile_body_25, video_path=video,
                                  threshold=0.1, BODY_PARTS=BODY_PARTS_BODY_25, POSE_PAIRS=POSE_PAIRS_BODY_25)
