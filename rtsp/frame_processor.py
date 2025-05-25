from ultralytics import YOLO
import torch
import numpy as np
import cv2
from sqlalchemy.orm import Session
from db.database import get_db
from db.crud import save_detection
import os
import time
from datetime import datetime, timedelta

# CUDA 디바이스 설정 (GPU가 사용 가능하면 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드 및 디바이스로 이동
# model = YOLO("models/yolo11n.pt")

# modelPerson = YOLO("models/yolo10s_person2.pt").to(device) # 사람 감지
modelPerson = YOLO("models/yolo11n.pt").to(device)
modelPose = YOLO("models/yolo11n-pose.pt").to(device)  # 포즈 탐지 모델 (Skeleton 등)
modelKisok = YOLO("models/kiosk.pt").to(device) #카드결제 - 카드감지, 결제감지 


STORE_ID = os.getenv("STORE_ID")
STORE_NAME = os.getenv("STORE_NAME")

CAMERA_ID1 = STORE_NAME+"_CCTV_1"
CAMERA_ID2 = STORE_NAME+"_CCTV_2"

def detect_person(frame, camera_id):
    """
    한 프레임에서 YOLO 객체 탐지(yolo11n)와 포즈 탐지(yolo11n-pos)를 동시에 실행.
    냉동고 위에 사람이 올라가 있는 상태를 감지하며, 5분 이상 지속 시 경고를 발생시킨다.
    또한, 모든 CCTV에서 100프레임 이상 감지가 되지 않으면 내려온 것으로 간주한다.
    """
    frame = cv2.resize(frame, (1024, 512))
    results_person = modelPerson(frame, conf=0.6)
    results_pose = modelPose(frame, conf=0.6)

    detections = []

    for result in results_person:
        for i, box in enumerate(result.boxes.xyxy):
            class_id = int(result.boxes.cls[i])
            x_min, y_min, x_max, y_max = map(float, box[:4])
            confidence = float(box[4]) if len(box) >= 5 else 0
            detections.append({
                "name": result.names[class_id],
                "confidence": confidence,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
            })

    left_hip_x = None
    right_hip_x = None

    if camera_id == CAMERA_ID1:
        fixed_line_x1, fixed_line_y1 = 270, 170
        fixed_line_x2, fixed_line_y2 = 190, 470
        fixed_line_x3, fixed_line_y3 = 393, 150
        fixed_line_x4, fixed_line_y4 = 555, 470
        cv2.line(frame, (fixed_line_x1, fixed_line_y1), (fixed_line_x2, fixed_line_y2), (0, 255, 0), 2)
        cv2.line(frame, (fixed_line_x3, fixed_line_y3), (fixed_line_x4, fixed_line_y4), (0, 255, 0), 2)
    elif camera_id == CAMERA_ID2:
        fixed_line_x1, fixed_line_y1 = 270, 170
        fixed_line_x2, fixed_line_y2 = 190, 470
        fixed_line_x3, fixed_line_y3 = 395, 170
        fixed_line_x4, fixed_line_y4 = 505, 470
        cv2.line(frame, (fixed_line_x1, fixed_line_y1), (fixed_line_x2, fixed_line_y2), (0, 255, 0), 2)
        cv2.line(frame, (fixed_line_x3, fixed_line_y3), (fixed_line_x4, fixed_line_y4), (0, 255, 0), 2)

    # 전역 상태 공유용 속성 초기화
    if not hasattr(detect_person, "global_alert_start_time"):
        detect_person.global_alert_start_time = None
    if not hasattr(detect_person, "global_alert_triggered"):
        detect_person.global_alert_triggered = False
    if not hasattr(detect_person, "global_missing_frame_count"):
        detect_person.global_missing_frame_count = 0

    crossed = False

    for result in results_pose:
        keypoints = result.keypoints
        for kp in keypoints.xy:
            xyList = kp.tolist()
            if len(xyList) < 13:
                continue

            left_hip_x = int(round(xyList[11][0]))
            right_hip_x = int(round(xyList[12][0]))
            cv2.circle(frame, (left_hip_x, int(round(xyList[11][1]))), 3, (0, 0, 255), -1)
            cv2.circle(frame, (right_hip_x, int(round(xyList[12][1]))), 3, (0, 0, 255), -1)

            if left_hip_x < fixed_line_x1 or left_hip_x > fixed_line_x3 or right_hip_x < fixed_line_x1 or right_hip_x > fixed_line_x3:
                crossed = True
                break

    if crossed:
        detect_person.global_missing_frame_count = 0
        if not detect_person.global_alert_start_time:
            detect_person.global_alert_start_time = datetime.now()
            detect_person.global_alert_triggered = False
        elif not detect_person.global_alert_triggered and datetime.now() - detect_person.global_alert_start_time >= timedelta(minutes=5):
            detect_person.global_alert_triggered = True
            print("[경고] 기물파손 위험감지 - 5분 이상 선을 넘은 상태")
    else:
        if detect_person.global_alert_start_time:
            detect_person.global_missing_frame_count += 1
            if detect_person.global_missing_frame_count >= 100:
                print("[정보] 사람 하강 감지 - 감지 해제")
                detect_person.global_alert_start_time = None
                detect_person.global_alert_triggered = False
                detect_person.global_missing_frame_count = 0

    for detection in detections:
        x_min, y_min, x_max, y_max = int(detection["x_min"]), int(detection["y_min"]), int(detection["x_max"]), int(detection["y_max"])
        label = f"{detection['name']} ({detection['confidence']:.2f})"
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    detected_count = len(detections)
    return frame, detected_count



# 전역 변수 선언 (함수 밖에서)
card_payment_active = False
missing_frame_counter = 0
detected_frame_counter = 0

def detect_paying(frame, camera_id):
    global card_payment_active, missing_frame_counter, detected_frame_counter  # 전역 변수 선언

    resized_frame = cv2.resize(frame, (1024, 512))
    results = modelKisok(resized_frame, conf=0.6)

    detections = []
    for result in results:
        for i, box in enumerate(result.boxes.xyxy):
            conf = float(box[4]) if len(box) >= 5 else 0
            class_id = int(result.boxes.cls[i])
            x_min = float(box[0])
            y_min = float(box[1])
            x_max = float(box[2])
            y_max = float(box[3])
            detections.append({
                "name": result.names[class_id],
                "confidence": conf,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
            })
    
    # # DB에 검출 결과 저장
    # db: Session = get_db()
    # save_detection(db, camera_id, detections)

    # 카드 감지 로직: 현재 프레임에서 "card" 클래스 감지 여부 확인 (대소문자 구분 없이)
    card_detected_in_frame = any(det["name"].lower() == "card" for det in detections)
    
    if card_detected_in_frame:
        detected_frame_counter += 1   # 카드 감지된 프레임 수 증가
        missing_frame_counter = 0       # 미감지 프레임 카운터 리셋
        # 10프레임 이상 연속 감지되었으면 결제 시작 처리
        if detected_frame_counter >= 10 and not card_payment_active:
            card_payment_active = True
            print("Card payment started.")
    else:
        detected_frame_counter = 0       # 카드가 감지되지 않으면 감지 카운터 초기화
        if card_payment_active:
            missing_frame_counter += 1   # 미감지 프레임 카운터 증가
            # 10프레임 이상 연속 감지 안되면 결제 종료 처리
            if missing_frame_counter >= 10:
                card_payment_active = False
                print("Card payment ended.")
    
    # 검출된 객체에 대해 경계 상자와 라벨 표시
    for detection in detections:
        x_min = int(detection["x_min"])
        y_min = int(detection["y_min"])
        x_max = int(detection["x_max"])
        y_max = int(detection["y_max"])
        label = detection["name"]
        cv2.rectangle(resized_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(resized_frame, label, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # 처리된 프레임 반환 (창 출력 및 키 입력 처리는 호출한 스레드에서 진행)
    return resized_frame


