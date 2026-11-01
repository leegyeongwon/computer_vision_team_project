"""
Squat Pose Analyzer
-------------------
기존 Detecting Pose.py에 관절 각도 계산 및 간단한 스쿼트 자세 분석을 추가한 통합 예제.

필요 패키지:
    pip install opencv-python mediapipe numpy

같은 폴더에 pose_landmarker_lite.task 파일이 있어야 합니다.

조작:
    b : 빈 배경 학습 완료 / 배경 고정
    m : 원본 영상과 MOG2 ROI 입력 전환 (배경 고정 후 가능)
    r : 배경 모델 초기화
    q : 종료

참고:
- 무릎 각도와 상체 기울기는 측면 촬영을 기준으로 한 간단한 규칙 기반 지표입니다.
- 각도 기준은 시연용 초기값이며, 사람·카메라 위치·운동 방식에 따라 조정해야 합니다.
- 이 프로그램은 의료적 평가나 부상 위험 진단을 목적으로 하지 않습니다.
"""

from pathlib import Path
import math

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ---------------------------------------------------------
# 1. 설정
# ---------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "pose_landmarker_lite.task"

VISIBILITY_THRESHOLD = 0.5

# 초기 테스트용 기준값
SQUAT_KNEE_MIN = 80.0
SQUAT_KNEE_MAX = 110.0
TORSO_TILT_MAX = 40.0

# MediaPipe landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

MAIN_JOINTS = {
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE
}


# ---------------------------------------------------------
# 2. 관절 각도 및 좌표 처리 함수
# ---------------------------------------------------------

def calculate_angle(a, b, c):
    """
    세 점 A-B-C에서 B를 중심으로 하는 각도를 계산한다.
    입력 좌표는 (x, y) 형태이다.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    vector_ba = a - b
    vector_bc = c - b

    norm_ba = np.linalg.norm(vector_ba)
    norm_bc = np.linalg.norm(vector_bc)

    if norm_ba == 0 or norm_bc == 0:
        return None

    cosine = np.dot(vector_ba, vector_bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine)))


def get_point(landmarks, landmark_id, frame_width, frame_height,
              offset_x=0, offset_y=0):
    """
    랜드마크의 정규화 좌표를 원본 영상의 픽셀 좌표로 변환한다.
    ROI를 사용한 경우 offset_x, offset_y를 더해 원본 영상 기준으로 맞춘다.
    """
    landmark = landmarks[landmark_id]

    if landmark.visibility < VISIBILITY_THRESHOLD:
        return None

    x = int(landmark.x * frame_width) + offset_x
    y = int(landmark.y * frame_height) + offset_y

    return (x, y)


def analyze_squat(landmarks, frame_width, frame_height,
                  offset_x=0, offset_y=0):
    """
    MediaPipe 랜드마크 한 사람 분량을 받아 스쿼트 자세를 분석한다.

    분석 항목:
      - 무릎 각도: 엉덩이-무릎-발목
      - 엉덩이 각도: 어깨-엉덩이-무릎
      - 상체 기울기: 수직선과 어깨-엉덩이 선 사이의 각도

    좌우 중 필요한 관절의 visibility 평균이 높은 쪽을 사용한다.
    """
    sides = {
        "LEFT": {
            "shoulder": LEFT_SHOULDER,
            "hip": LEFT_HIP,
            "knee": LEFT_KNEE,
            "ankle": LEFT_ANKLE
        },
        "RIGHT": {
            "shoulder": RIGHT_SHOULDER,
            "hip": RIGHT_HIP,
            "knee": RIGHT_KNEE,
            "ankle": RIGHT_ANKLE
        }
    }

    candidates = []

    for side_name, ids in sides.items():
        visibility_values = [
            landmarks[index].visibility for index in ids.values()
        ]

        if min(visibility_values) < VISIBILITY_THRESHOLD:
            continue

        points = {
            name: get_point(
                landmarks, landmark_id,
                frame_width, frame_height,
                offset_x, offset_y
            )
            for name, landmark_id in ids.items()
        }

        if any(point is None for point in points.values()):
            continue

        average_visibility = sum(visibility_values) / len(visibility_values)
        candidates.append((average_visibility, side_name, points))

    if not candidates:
        return {
            "valid": False,
            "message": "관절이 충분히 보이도록 전신을 촬영하세요."
        }

    # 신뢰도가 높은 쪽 선택
    _, selected_side, points = max(candidates, key=lambda item: item[0])

    knee_angle = calculate_angle(
        points["hip"], points["knee"], points["ankle"]
    )
    hip_angle = calculate_angle(
        points["shoulder"], points["hip"], points["knee"]
    )

    # 어깨-엉덩이 선과 화면의 수직선 사이의 기울기
    dx = points["shoulder"][0] - points["hip"][0]
    dy = points["shoulder"][1] - points["hip"][1]

    if dx == 0 and dy == 0:
        torso_tilt = None
    else:
        torso_tilt = math.degrees(math.atan2(abs(dx), abs(dy)))

    if knee_angle is None or hip_angle is None or torso_tilt is None:
        return {
            "valid": False,
            "message": "각도를 계산할 수 없습니다."
        }

    # 무릎 각도를 이용한 간단한 동작 구간 판별
    if knee_angle <= SQUAT_KNEE_MAX:
        squat_phase = "DOWN / SQUAT"
    elif knee_angle >= 150:
        squat_phase = "STANDING"
    else:
        squat_phase = "MOVING"

    # 초기 기준에 따른 간단한 자세 조건 검사
    depth_ok = SQUAT_KNEE_MIN <= knee_angle <= SQUAT_KNEE_MAX
    torso_ok = torso_tilt <= TORSO_TILT_MAX

    feedback = []

    if squat_phase == "DOWN / SQUAT" and not depth_ok:
        feedback.append("무릎 각도를 목표 범위와 비교하세요.")

    if not torso_ok:
        feedback.append("상체가 많이 앞으로 기울어져 있습니다.")

    if not feedback:
        feedback.append("분석 기준 범위 안에 있습니다.")

    return {
        "valid": True,
        "side": selected_side,
        "points": points,
        "knee_angle": knee_angle,
        "hip_angle": hip_angle,
        "torso_tilt": torso_tilt,
        "squat_phase": squat_phase,
        "depth_ok": depth_ok,
        "torso_ok": torso_ok,
        "feedback": feedback
    }


# ---------------------------------------------------------
# 3. MediaPipe PoseLandmarker 초기화
# ---------------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}\n"
        "pose_landmarker_lite.task 파일을 이 파이썬 파일과 같은 폴더에 두세요."
    )

base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    num_poses=1
)
pose_detector = vision.PoseLandmarker.create_from_options(options)

connections = vision.PoseLandmarksConnections.POSE_LANDMARKS


# ---------------------------------------------------------
# 4. 웹캠 및 배경 분리 초기화
# ---------------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    pose_detector.close()
    raise SystemExit("웹캠을 열 수 없습니다. 카메라 연결과 권한을 확인하세요.")

background_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=False
)

background_ready = False
use_mog2 = False

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# ---------------------------------------------------------
# 5. 실시간 영상 처리
# ---------------------------------------------------------

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("웹캠 프레임을 읽을 수 없습니다.")
            break

        # 전면 카메라처럼 보이도록 좌우 반전
        frame = cv2.flip(frame, 1)

        # 배경 마스크 생성
        if background_ready:
            foreground_mask = background_subtractor.apply(
                frame, learningRate=0
            )
        else:
            foreground_mask = background_subtractor.apply(
                frame, learningRate=-1
            )

        foreground_mask = cv2.morphologyEx(
            foreground_mask, cv2.MORPH_CLOSE, kernel
        )

        height, width = frame.shape[:2]

        # 기본적으로 전체 프레임 사용
        x1, y1, x2, y2 = 0, 0, width, height
        input_frame = frame.copy()
        input_mode = "Original"

        # MOG2가 활성화된 경우 사람 영역을 추정해 ROI 입력
        if use_mog2 and background_ready:
            contours, _ = cv2.findContours(
                foreground_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            boxes = [
                cv2.boundingRect(contour)
                for contour in contours
                if cv2.contourArea(contour) > width * height * 0.001
            ]

            if boxes:
                x1 = min(box[0] for box in boxes)
                y1 = min(box[1] for box in boxes)
                x2 = max(box[0] + box[2] for box in boxes)
                y2 = max(box[1] + box[3] for box in boxes)

                margin_x = int((x2 - x1) * 0.2)
                margin_y = int((y2 - y1) * 0.2)

                x1 = max(0, x1 - margin_x)
                y1 = max(0, y1 - margin_y)
                x2 = min(width, x2 + margin_x)
                y2 = min(height, y2 + margin_y)

                input_frame = frame[y1:y2, x1:x2].copy()
                input_mode = "MOG2 ROI"
            else:
                input_mode = "Original (no ROI)"

        input_height, input_width = input_frame.shape[:2]
        rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )
        result = pose_detector.detect(mp_image)

        # ROI 영역 표시
        if input_mode == "MOG2 ROI":
            cv2.rectangle(
                frame, (x1, y1), (x2 - 1, y2 - 1),
                (0, 255, 255), 2
            )

        cv2.putText(
            frame,
            f"Input: {input_mode} | Poses: {len(result.pose_landmarks)}",
            (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 255, 255), 2
        )

        # 검출된 사람별 랜드마크 표시 및 자세 분석
        for pose_landmarks in result.pose_landmarks:

            # ROI 좌표를 원본 영상 좌표로 변환
            points = [
                (
                    int(point.x * input_width) + x1,
                    int(point.y * input_height) + y1
                )
                for point in pose_landmarks
            ]

            # 관절 연결선
            for connection in connections:
                start = connection.start
                end = connection.end

                if (
                    pose_landmarks[start].visibility >= VISIBILITY_THRESHOLD
                    and pose_landmarks[end].visibility >= VISIBILITY_THRESHOLD
                ):
                    cv2.line(
                        frame, points[start], points[end],
                        (0, 255, 0), 2
                    )

            # 관절 점
            for index, point in enumerate(points):
                if pose_landmarks[index].visibility < VISIBILITY_THRESHOLD:
                    continue

                color = (0, 0, 255) if index in MAIN_JOINTS else (255, 0, 0)
                radius = 6 if index in MAIN_JOINTS else 4
                cv2.circle(frame, point, radius, color, -1)

            # 네 담당 기능: 각도 계산 및 자세 분석
            analysis = analyze_squat(
                pose_landmarks,
                input_width,
                input_height,
                offset_x=x1,
                offset_y=y1
            )

            if analysis["valid"]:
                # 분석에 사용한 관절 위치에 각도 표시
                knee_point = analysis["points"]["knee"]
                hip_point = analysis["points"]["hip"]

                cv2.putText(
                    frame,
                    f"Knee: {analysis['knee_angle']:.1f} deg",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 0), 2
                )

                cv2.putText(
                    frame,
                    f"Hip: {analysis['hip_angle']:.1f} deg",
                    (20, 105), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 0), 2
                )

                cv2.putText(
                    frame,
                    f"Torso tilt: {analysis['torso_tilt']:.1f} deg",
                    (20, 135), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 0), 2
                )

                cv2.putText(
                    frame,
                    f"State: {analysis['squat_phase']}",
                    (20, 170), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 255), 2
                )

                # 간단한 피드백 표시
                feedback_color = (
                    (0, 255, 0)
                    if analysis["torso_ok"]
                    else (0, 0, 255)
                )

                cv2.putText(
                    frame,
                    analysis["feedback"][0],
                    (20, 205), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, feedback_color, 2
                )

                # 무릎과 엉덩이 각도 표시 위치
                cv2.putText(
                    frame, f"{analysis['knee_angle']:.0f}",
                    (knee_point[0] + 10, knee_point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 0), 2
                )
                cv2.putText(
                    frame, f"{analysis['hip_angle']:.0f}",
                    (hip_point[0] + 10, hip_point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 0), 2
                )

            else:
                cv2.putText(
                    frame,
                    analysis["message"],
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2
                )

        # 자세 미검출 안내
        if not result.pose_landmarks:
            cv2.putText(
                frame,
                "No pose detected. Show your full body.",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2
            )

        bg_status = "Fixed" if background_ready else "Learning"

        cv2.putText(
            frame,
            f"BG: {bg_status} | b: fix | m: mode | r: reset | q: quit",
            (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 255), 1
        )

        cv2.imshow("Squat Pose Analyzer", frame)
        cv2.imshow("MOG2 Foreground Mask", foreground_mask)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("b"):
            background_ready = True
            print("Background fixed. You can enter the camera view.")

        elif key == ord("m"):
            if background_ready:
                use_mog2 = not use_mog2
            else:
                print("먼저 빈 배경을 학습한 후 b를 누르세요.")

        elif key == ord("r"):
            background_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500,
                varThreshold=16,
                detectShadows=False
            )
            background_ready = False
            use_mog2 = False
            print("Background reset.")

        elif key == ord("q"):
            break

finally:
    pose_detector.close()
    cap.release()
    cv2.destroyAllWindows()
