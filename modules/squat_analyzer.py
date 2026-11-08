"""
squat_session.py
----------------
자세 감지 모듈. 웹캠, MediaPipe, MOG2, 창 표시는 main.py가 담당한다.

사용법 (main.py):
    from squat_session import (
        process_squat, toggle_panel, new_set, finish_squat_session
    )

    def handle_frame_data(data):
        analysis = process_squat(data)
        ...

    # 키 처리
    elif key == ord("s"): toggle_panel()
    elif key == ord("x"): new_set()

    # 종료 직전 (cv2.destroyAllWindows() 호출 전에)
    finish_squat_session()

필요 파일:
    feedback_record.py (SquatTracker)

참고:
- 무릎 각도와 상체 기울기는 측면 촬영을 기준으로 한 2D 화면 좌표 기반 지표입니다.
- 기준값은 시연용 초기값이며, 사람·카메라 위치·운동 방식에 따라 조정해야 합니다.
- 의료적 평가나 부상 위험 진단을 목적으로 하지 않습니다.
"""

import math
from pathlib import Path

import cv2
import numpy as np

from .feedback_record import SquatTracker


# ---------------------------------------------------------
# 1. 설정
# ---------------------------------------------------------

VISIBILITY_THRESHOLD = 0.5

# 초기 테스트용 기준값 (도)
SQUAT_KNEE_MIN = 80.0
SQUAT_KNEE_MAX = 110.0
STANDING_KNEE_MIN = 150.0
TORSO_TILT_MAX = 40.0

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

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_INFO = (255, 255, 0)
COLOR_STATE = (0, 255, 255)
COLOR_WARN = (0, 0, 255)

# 피드백 및 운동 기록 모듈 (import 시 한 번 생성)
tracker = SquatTracker(
    knee_min=SQUAT_KNEE_MIN,
    knee_max=SQUAT_KNEE_MAX,
    torso_max=TORSO_TILT_MAX,
    save_dir=Path(__file__).parent / "records",
)


# ---------------------------------------------------------
# 2. 각도 계산
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
# 3. 스쿼트 분석
# ---------------------------------------------------------

def analyze_squat(landmarks, visibility_threshold=VISIBILITY_THRESHOLD):
    """
    한 사람의 랜드마크로 스쿼트 자세를 분석한다.

    landmarks: frame_data["landmarks_px"][pose]
               landmarks[idx] = (x_px, y_px, z, visibility), 원본 프레임 픽셀 좌표

    반환 dict 형식은 squat_pose_analyzer.py와 같다 (SquatTracker 호환).
      valid=False -> "message"만 있음
      valid=True  -> side, points, knee_angle, hip_angle, torso_tilt,
                     squat_phase, depth_ok, torso_ok, feedback(문자열 리스트)
    experiment_logger 등에서 쓰기 쉽도록 "phase"는 "squat_phase"의 별칭으로 함께 넣는다.
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
            "message": "관절이 충분히 보이도록 전신을 촬영하세요.",
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
        return {"valid": False, "message": "각도를 계산할 수 없습니다."}

    # 무릎 각도 기반 동작 구간
    if knee_angle <= SQUAT_KNEE_MAX:
        squat_phase = "DOWN / SQUAT"
    elif knee_angle >= STANDING_KNEE_MIN:
        squat_phase = "STANDING"
    else:
        squat_phase = "MOVING"

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
        "side": side,
        "points": points,
        "knee_angle": knee_angle,
        "hip_angle": hip_angle,
        "torso_tilt": torso_tilt,
        "squat_phase": squat_phase,
        "phase": squat_phase,       # 별칭
        "depth_ok": depth_ok,
        "torso_ok": torso_ok,
        "feedback": feedback,
    }


# ---------------------------------------------------------
# 4. 화면 표시
# ---------------------------------------------------------

def draw_angle_overlay(frame, analysis, origin=(20, 75)):
    """
    각도/상태 텍스트를 frame 위에 직접 그린다. (frame은 제자리에서 수정됨)
    피드백 문구와 관절 강조는 tracker.draw()가 그린다.
    """
    x, y = origin

    if not analysis["valid"]:
        # analysis["message"]는 한글이라 cv2.putText로 표시할 수 없다.
        cv2.putText(frame, "Joints not visible. Show full body (side view).",
                    (x, y), FONT, 0.6, COLOR_WARN, 2)
        return

    cv2.putText(frame, f"Knee: {analysis['knee_angle']:.1f} deg",
                (x, y), FONT, 0.65, COLOR_INFO, 2)
    cv2.putText(frame, f"Hip: {analysis['hip_angle']:.1f} deg",
                (x, y + 30), FONT, 0.65, COLOR_INFO, 2)
    cv2.putText(frame, f"Torso tilt: {analysis['torso_tilt']:.1f} deg",
                (x, y + 60), FONT, 0.65, COLOR_INFO, 2)
    cv2.putText(frame, f"State: {analysis['squat_phase']} ({analysis['side']})",
                (x, y + 95), FONT, 0.65, COLOR_STATE, 2)

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
# 5. main.py에서 부르는 진입점
# ---------------------------------------------------------

def process_squat(data):
    """
    frame_data 한 프레임을 받아 분석하고, 기록을 갱신하고, annotated_frame에 그린다.
    분석 결과 dict를 반환한다. (감지 단계가 아니거나 사람이 없으면 None)
    """
    if data["stage"] != "detect":
        return None

    analysis = None
    if data["num_poses"] > 0:
        analysis = analyze_squat(data["landmarks_px"][0])
        draw_angle_overlay(data["annotated_frame"], analysis)

    # 사람이 없는 프레임(None)도 tracker에 넘긴다 (원본과 동일)
    tracker.update(analysis)
    tracker.draw(data["annotated_frame"], analysis)

    return analysis


# ---------------------------------------------------------
# 6. 키 입력 / 종료 처리 (main.py의 키 분기에서 호출)
# ---------------------------------------------------------

def toggle_panel():
    """s: 운동 기록 패널 표시/숨김"""
    tracker.show_panel = not tracker.show_panel


def new_set():
    """x: 현재 세트를 저장하고 새 세트를 시작한다."""
    tracker.print_summary()
    saved = tracker.save_summary()
    if saved:
        print(f"기록 저장: {saved}")
    tracker.new_session()
    print("새 세트를 시작합니다.")


def finish_squat_session(show_result=True):
    """
    종료 시 호출한다. 결과 화면을 보여 주고(아무 키나 누르면 닫힘) 기록을 저장한다.
    cv2.destroyAllWindows() 보다 먼저 불러야 결과 창이 보인다.
    """
    if show_result:
        cv2.imshow("Squat Session Result", tracker.render_summary_image())
        cv2.waitKey(0)

    tracker.print_summary()
    saved = tracker.save_summary()
    if saved:
        print(f"기록 저장: {saved}")