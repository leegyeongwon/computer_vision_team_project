"""
squat_analyzer.py
-----------------
메인 파이프라인(frame_data)에 붙여 쓰는 스쿼트 자세 분석 모듈.

사용법 (메인 코드의 handle_frame_data 안에서):
    from squat_analyzer import process_squat

    def handle_frame_data(data):
        process_squat(data)

필요 패키지:
    pip install opencv-python numpy

참고:
- 무릎 각도와 상체 기울기는 측면 촬영을 기준으로 한 2D 화면 좌표 기반 지표입니다.
- 기준값은 시연용 초기값이며, 사람·카메라 위치·운동 방식에 따라 조정해야 합니다.
- 의료적 평가나 부상 위험 진단을 목적으로 하지 않습니다.
"""

import math

import cv2
import numpy as np


# ---------------------------------------------------------
# 설정
# ---------------------------------------------------------

VISIBILITY_THRESHOLD = 0.5

# 초기 테스트용 기준값 (도)
SQUAT_KNEE_MAX = 110.0     # 이 값 이하이면 DOWN(스쿼트 깊이 도달)
STANDING_KNEE_MIN = 150.0  # 이 값 이상이면 STANDING
TORSO_TILT_MAX = 40.0      # 이 값을 넘으면 상체가 많이 기울어진 것으로 판단

# MediaPipe landmark indices
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

SIDES = {
    "LEFT": {
        "shoulder": LEFT_SHOULDER,
        "hip": LEFT_HIP,
        "knee": LEFT_KNEE,
        "ankle": LEFT_ANKLE,
    },
    "RIGHT": {
        "shoulder": RIGHT_SHOULDER,
        "hip": RIGHT_HIP,
        "knee": RIGHT_KNEE,
        "ankle": RIGHT_ANKLE,
    },
}

# 화면 표시용 (BGR). 
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_INFO = (255, 255, 0)
COLOR_STATE = (0, 255, 255)
COLOR_OK = (0, 255, 0)
COLOR_WARN = (0, 0, 255)


# ---------------------------------------------------------
# 각도 계산
# ---------------------------------------------------------

def calculate_angle(a, b, c):
    """세 점 A-B-C에서 B를 중심으로 하는 각도(도)를 계산한다. (x, y) 입력."""
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


# ---------------------------------------------------------
# 스쿼트 분석
# ---------------------------------------------------------

def analyze_squat(landmarks, visibility_threshold=VISIBILITY_THRESHOLD):
    """
    한 사람의 랜드마크로 스쿼트 자세를 분석한다.

    landmarks: frame_data["landmarks_px"][pose]
               landmarks[idx] = (x_px, y_px, z, visibility), 원본 프레임 픽셀 좌표

    좌우 중 필요한 관절이 모두 보이는 쪽(visibility 평균이 높은 쪽)을 사용한다.

    반환: dict
      valid=False 이면 "message"만 있음.
      valid=True 이면 side, points, knee_angle, hip_angle, torso_tilt,
                      phase, torso_ok, feedback([(문구, 정상여부), ...])
    """
    candidates = []

    for side_name, ids in SIDES.items():
        visibilities = [landmarks[i][3] for i in ids.values()]

        if min(visibilities) < visibility_threshold:
            continue

        points = {
            name: (int(landmarks[i][0]), int(landmarks[i][1]))
            for name, i in ids.items()
        }
        average_visibility = sum(visibilities) / len(visibilities)
        candidates.append((average_visibility, side_name, points))

    if not candidates:
        return {
            "valid": False,
            "message": "Joints not visible. Show full body from the side.",
        }

    _, side, points = max(candidates, key=lambda item: item[0])

    knee_angle = calculate_angle(points["hip"], points["knee"], points["ankle"])
    hip_angle = calculate_angle(points["shoulder"], points["hip"], points["knee"])

    # 어깨-엉덩이 선과 화면 수직선 사이의 기울기
    dx = points["shoulder"][0] - points["hip"][0]
    dy = points["shoulder"][1] - points["hip"][1]
    torso_tilt = None if (dx == 0 and dy == 0) else math.degrees(
        math.atan2(abs(dx), abs(dy))
    )

    if knee_angle is None or hip_angle is None or torso_tilt is None:
        return {"valid": False, "message": "Cannot compute angles."}

    # 무릎 각도 기반 동작 구간
    if knee_angle <= SQUAT_KNEE_MAX:
        phase = "DOWN"
    elif knee_angle >= STANDING_KNEE_MIN:
        phase = "STANDING"
    else:
        phase = "MOVING"

    torso_ok = torso_tilt <= TORSO_TILT_MAX

    feedback = []
    if phase == "DOWN":
        feedback.append(("Depth reached", True))
    if not torso_ok:
        feedback.append(("Torso leaning forward too much", False))
    if not feedback:
        feedback.append(("Within reference range", True))

    return {
        "valid": True,
        "side": side,
        "points": points,
        "knee_angle": knee_angle,
        "hip_angle": hip_angle,
        "torso_tilt": torso_tilt,
        "phase": phase,
        "torso_ok": torso_ok,
        "feedback": feedback,
    }


# ---------------------------------------------------------
# 화면 표시
# ---------------------------------------------------------

def draw_squat_overlay(frame, analysis, origin=(20, 75)):
    """분석 결과를 frame 위에 직접 그린다. (frame은 제자리에서 수정됨)"""
    x, y = origin

    if not analysis["valid"]:
        cv2.putText(frame, analysis["message"], (x, y),
                    FONT, 0.6, COLOR_WARN, 2)
        return

    cv2.putText(frame, f"Knee: {analysis['knee_angle']:.1f} deg",
                (x, y), FONT, 0.65, COLOR_INFO, 2)
    cv2.putText(frame, f"Hip: {analysis['hip_angle']:.1f} deg",
                (x, y + 30), FONT, 0.65, COLOR_INFO, 2)
    cv2.putText(frame, f"Torso tilt: {analysis['torso_tilt']:.1f} deg",
                (x, y + 60), FONT, 0.65, COLOR_INFO, 2)
    cv2.putText(frame, f"State: {analysis['phase']} ({analysis['side']})",
                (x, y + 95), FONT, 0.65, COLOR_STATE, 2)

    # 피드백은 모두 표시하고, 항목마다 색을 따로 정한다.
    for i, (text, ok) in enumerate(analysis["feedback"]):
        cv2.putText(frame, text, (x, y + 130 + 28 * i),
                    FONT, 0.55, COLOR_OK if ok else COLOR_WARN, 2)

    # 관절 옆에 각도 숫자 표시
    knee_point = analysis["points"]["knee"]
    hip_point = analysis["points"]["hip"]
    cv2.putText(frame, f"{analysis['knee_angle']:.0f}",
                (knee_point[0] + 10, knee_point[1]),
                FONT, 0.55, COLOR_INFO, 2)
    cv2.putText(frame, f"{analysis['hip_angle']:.0f}",
                (hip_point[0] + 10, hip_point[1]),
                FONT, 0.55, COLOR_INFO, 2)


# ---------------------------------------------------------
# 메인 코드에서 부르는 진입점
# ---------------------------------------------------------

def process_squat(data):
    """
    frame_data 한 프레임을 받아 분석하고 annotated_frame에 결과를 그린다.
    분석 결과 dict를 반환한다. (감지 단계가 아니거나 사람이 없으면 None)
    """
    if data["stage"] != "detect" or data["num_poses"] == 0:
        return None

    analysis = analyze_squat(data["landmarks_px"][0])
    draw_squat_overlay(data["annotated_frame"], analysis)
    return analysis