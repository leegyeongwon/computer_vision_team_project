"""
Feedback & Exercise Record (피드백 및 운동 기록 기능)
-----------------------------------------------------
analyze_squat()가 반환하는 분석 결과(dict)를 입력으로 받아
  1) 반복(rep) 횟수 카운트
  2) 반복별 자세 판정 (기준 충족 / 이탈 + 이탈 유형)
  3) 자세 이탈 관절 강조 표시 + 화면 안내 메시지
  4) 운동 기록 저장 (반복별 CSV, 세션 요약 JSON)
  5) 실시간 통계 패널 및 운동 종료 후 결과 화면
을 담당한다.

입력 데이터 형식 (팀 공통 인터페이스 = analyze_squat 반환값):
    {
      "valid": bool,
      "points": {"shoulder": (x, y), "hip": (x, y), "knee": (x, y), "ankle": (x, y)},
      "knee_angle": float, "hip_angle": float, "torso_tilt": float, ...
    }

사용법:
    tracker = SquatTracker(knee_min=80, knee_max=110, torso_max=40)
    # 매 프레임
    tracker.update(analysis)          # 사람이 없으면 None
    tracker.draw(frame, analysis)
    # 종료 시
    tracker.print_summary(); tracker.save_summary()

참고: cv2.putText는 한글을 그리지 못하므로 화면 문구는 영어,
      콘솔 출력과 저장 파일은 한글을 사용한다.
"""

from __future__ import annotations

import csv
import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------
# 설정값
# ---------------------------------------------------------

STAND_KNEE = 150.0        # 무릎 각도가 이 이상이면 선 자세
DESCEND_KNEE = 140.0      # 선 자세에서 이 아래로 내려가면 반복 시작 (150과 차이를 둬 떨림 방지)
SMOOTH_WINDOW = 5         # 각도 이동평균 프레임 수
LOST_FRAMES_LIMIT = 15    # 연속으로 이만큼 분석 실패 시 진행 중인 반복 취소
MIN_REP_SEC = 0.5         # 이보다 짧은 반복은 노이즈로 보고 무시
MESSAGE_HOLD_SEC = 1.5    # 반복 판정 메시지 유지 시간

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 200, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)

# 이탈 유형: (화면 문구, 한글 설명)
ERROR_INFO = {
    "SHALLOW": ("Go deeper", "앉는 깊이 부족"),
    "TOO_DEEP": ("Too deep", "기준보다 깊게 앉음"),
    "TORSO_LEAN": ("Keep your chest up", "상체가 과도하게 기울어짐"),
}

# 이탈 유형별로 강조할 관절 (선으로 이어서 표시)
ERROR_JOINTS = {
    "SHALLOW": ("hip", "knee", "ankle"),
    "TOO_DEEP": ("hip", "knee", "ankle"),
    "TORSO_LEAN": ("shoulder", "hip"),
}


@dataclass
class RepRecord:
    rep: int
    time: str
    duration_sec: float
    min_knee: float
    min_hip: float
    max_torso: float
    good: bool
    errors: list = field(default_factory=list)


def _dark_box(frame, x0, y0, x1, y1, alpha=0.6):
    """반투명 검은 배경 박스."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    frame[y0:y1, x0:x1] = (roi * (1 - alpha)).astype(np.uint8)


class SquatTracker:
    def __init__(self, knee_min, knee_max, torso_max,
                 save_dir="records", save=True):
        self.knee_min = knee_min
        self.knee_max = knee_max
        self.torso_max = torso_max
        self.save_dir = Path(save_dir)
        self.save = save
        self.show_panel = True
        self.new_session()

    # -----------------------------------------------------
    # 세션 관리
    # -----------------------------------------------------

    def new_session(self):
        """기록을 비우고 새 세트를 시작한다."""
        self.session_start = time.time()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.records: list[RepRecord] = []
        self.state = "WAIT_STAND"   # WAIT_STAND -> STANDING -> DESCENDING -> BOTTOM -> STANDING
        self._rep = None
        self._lost = 0
        self._bufs = {k: deque(maxlen=SMOOTH_WINDOW)
                      for k in ("knee", "hip", "torso")}
        self.angles = None
        self.live_issues = []
        self._csv_path = None
        self._hold("Stand sideways, full body in view", YELLOW, [],
                   time.time(), sec=3.0)

    # -----------------------------------------------------
    # 프레임 단위 갱신
    # -----------------------------------------------------

    def _smooth(self, key, value):
        buf = self._bufs[key]
        buf.append(value)
        return sum(buf) / len(buf)

    def _hold(self, text, color, issues, now, sec=MESSAGE_HOLD_SEC):
        self._held_text = text
        self._held_color = color
        self._held_issues = list(issues)
        self._held_until = now + sec

    def update(self, analysis):
        """매 프레임 호출. analysis는 analyze_squat() 결과 또는 None."""
        now = time.time()
        self.live_issues = []

        if analysis is None or not analysis.get("valid"):
            self._lost += 1
            if self._lost > LOST_FRAMES_LIMIT:
                for buf in self._bufs.values():
                    buf.clear()
                self.angles = None
                if self._rep is not None:
                    self._rep = None
                    self.state = "WAIT_STAND"
                    self._hold("Tracking lost - rep cancelled", RED, [], now)
            return

        self._lost = 0
        knee = self._smooth("knee", analysis["knee_angle"])
        hip = self._smooth("hip", analysis["hip_angle"])
        torso = self._smooth("torso", analysis["torso_tilt"])
        self.angles = {"knee": knee, "hip": hip, "torso": torso}

        # 실시간 이탈 감지 (반복 진행 중일 때만 경고)
        if self.state in ("DESCENDING", "BOTTOM"):
            if torso > self.torso_max:
                self.live_issues.append("TORSO_LEAN")
            if knee < self.knee_min:
                self.live_issues.append("TOO_DEEP")

        # 반복 카운트 상태 머신
        if self.state == "WAIT_STAND":
            if knee >= STAND_KNEE:
                self.state = "STANDING"

        elif self.state == "STANDING":
            if knee < DESCEND_KNEE:
                self.state = "DESCENDING"
                self._rep = {"start": now, "min_knee": knee,
                             "min_hip": hip, "max_torso": torso}

        else:  # DESCENDING / BOTTOM
            rep = self._rep
            rep["min_knee"] = min(rep["min_knee"], knee)
            rep["min_hip"] = min(rep["min_hip"], hip)
            rep["max_torso"] = max(rep["max_torso"], torso)

            if knee <= self.knee_max:
                self.state = "BOTTOM"

            if knee >= STAND_KNEE:
                self._finish_rep(now)
                self.state = "STANDING"

    def _finish_rep(self, now):
        rep, self._rep = self._rep, None
        duration = now - rep["start"]
        if duration < MIN_REP_SEC:
            return

        errors = []
        if rep["min_knee"] > self.knee_max:
            errors.append("SHALLOW")
        elif rep["min_knee"] < self.knee_min:
            errors.append("TOO_DEEP")
        if rep["max_torso"] > self.torso_max:
            errors.append("TORSO_LEAN")

        record = RepRecord(
            rep=len(self.records) + 1,
            time=datetime.now().strftime("%H:%M:%S"),
            duration_sec=round(duration, 2),
            min_knee=round(rep["min_knee"], 1),
            min_hip=round(rep["min_hip"], 1),
            max_torso=round(rep["max_torso"], 1),
            good=not errors,
            errors=errors,
        )
        self.records.append(record)
        self._write_csv(record)

        if record.good:
            self._hold(f"Rep {record.rep}: GOOD!", GREEN, [], now)
            print(f"[{record.rep}회] 기준 충족 "
                  f"(무릎 최소 {record.min_knee}°, 상체 최대 {record.max_torso}°)")
        else:
            text = " / ".join(ERROR_INFO[e][0] for e in errors)
            self._hold(f"Rep {record.rep}: {text}", RED, errors, now)
            reasons = ", ".join(ERROR_INFO[e][1] for e in errors)
            print(f"[{record.rep}회] 기준 이탈 - {reasons} "
                  f"(무릎 최소 {record.min_knee}°, 상체 최대 {record.max_torso}°)")

    # -----------------------------------------------------
    # 화면 표시
    # -----------------------------------------------------

    def draw(self, frame, analysis=None):
        now = time.time()
        holding = now < self._held_until

        issues = list(self.live_issues)
        if holding:
            issues += [e for e in self._held_issues if e not in issues]

        if analysis and analysis.get("valid") and issues:
            self._draw_highlights(frame, analysis["points"], issues)

        if holding:
            text, color = self._held_text, self._held_color
        elif self.live_issues:
            text, color = ERROR_INFO[self.live_issues[0]][0], RED
        elif self.state == "WAIT_STAND":
            text, color = "Stand up straight to start", YELLOW
        elif self.state == "STANDING":
            text, color = "Ready - squat down", WHITE
        elif self.state == "BOTTOM":
            text, color = "Good depth - stand up", GREEN
        else:
            text, color = "Lower...", WHITE

        self._draw_banner(frame, text, color)
        if self.show_panel:
            self._draw_panel(frame)

    def _draw_highlights(self, frame, points, issues):
        """기준을 벗어난 신체 부위를 빨간 굵은 선과 원으로 강조."""
        for issue in issues:
            joints = [points.get(j) for j in ERROR_JOINTS[issue]]
            if any(p is None for p in joints):
                continue
            for a, b in zip(joints, joints[1:]):
                cv2.line(frame, a, b, RED, 6)
            for p in joints:
                cv2.circle(frame, p, 14, RED, 3)

    def _draw_banner(self, frame, text, color):
        h, w = frame.shape[:2]
        scale, thick = 0.9, 2
        (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
        x = max(10, (w - tw) // 2)
        y = h - 50
        _dark_box(frame, x - 10, y - th - 10, x + tw + 10, y + base + 6, 0.7)
        cv2.putText(frame, text, (x, y), FONT, scale, color, thick)

    def _stat_lines(self):
        s = self.summary()
        ec = s["error_counts"]
        m, sec = divmod(int(s["elapsed_sec"]), 60)
        depth = f"{s['avg_min_knee']:.0f} deg" if s["avg_min_knee"] is not None else "-"
        return [
            ("[ RECORD ]", YELLOW),
            (f"Reps: {s['total_reps']}", WHITE),
            (f"Good: {s['good_reps']}   Bad: {s['bad_reps']}", WHITE),
            (f"Accuracy: {s['accuracy']:.0f}%", GREEN if s["accuracy"] >= 80 else WHITE),
            (f"Shallow: {ec['SHALLOW']}  Deep: {ec['TOO_DEEP']}", WHITE),
            (f"Torso lean: {ec['TORSO_LEAN']}", WHITE),
            (f"Avg depth: {depth}", WHITE),
            (f"Time: {m:02d}:{sec:02d}", WHITE),
        ]

    def _draw_panel(self, frame):
        lines = self._stat_lines()
        w = 230
        h = 22 * len(lines) + 14
        x0 = frame.shape[1] - w - 10
        y0 = 50
        _dark_box(frame, x0, y0, x0 + w, y0 + h)
        for i, (text, color) in enumerate(lines):
            cv2.putText(frame, text, (x0 + 10, y0 + 24 + 22 * i),
                        FONT, 0.55, color, 1)

    def render_summary_image(self, width=640, height=480):
        """운동 종료 후 보여줄 결과 화면."""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        s = self.summary()
        cv2.putText(img, "SQUAT SESSION RESULT", (30, 50), FONT, 1.0, YELLOW, 2)
        y = 100
        for text, color in self._stat_lines()[1:]:
            cv2.putText(img, text, (40, y), FONT, 0.75, color, 2)
            y += 38

        # 최근 반복 결과를 막대로 표시 (초록 = 충족, 빨강 = 이탈)
        recent = self.records[-20:]
        if recent:
            cv2.putText(img, "Recent reps:", (40, y + 10), FONT, 0.6, WHITE, 1)
            for i, r in enumerate(recent):
                x = 40 + i * 28
                cv2.rectangle(img, (x, y + 25), (x + 20, y + 55),
                              GREEN if r.good else RED, -1)
        if s["total_reps"] == 0:
            cv2.putText(img, "No reps recorded.", (40, y + 10), FONT, 0.7, WHITE, 1)
        cv2.putText(img, "Press any key to close", (30, height - 20),
                    FONT, 0.5, (180, 180, 180), 1)
        return img

    # -----------------------------------------------------
    # 통계 및 저장
    # -----------------------------------------------------

    def summary(self):
        total = len(self.records)
        good = sum(r.good for r in self.records)

        def avg(attr):
            if not total:
                return None
            return round(sum(getattr(r, attr) for r in self.records) / total, 1)

        return {
            "session_id": self.session_id,
            "elapsed_sec": round(time.time() - self.session_start, 1),
            "total_reps": total,
            "good_reps": good,
            "bad_reps": total - good,
            "accuracy": round(100 * good / total, 1) if total else 0.0,
            "error_counts": {c: sum(c in r.errors for r in self.records)
                             for c in ERROR_INFO},
            "avg_min_knee": avg("min_knee"),
            "avg_max_torso": avg("max_torso"),
            "avg_rep_sec": avg("duration_sec"),
            "thresholds": {"knee_min": self.knee_min,
                           "knee_max": self.knee_max,
                           "torso_max": self.torso_max},
        }

    def _write_csv(self, record):
        if not self.save:
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if self._csv_path is None:
            self._csv_path = self.save_dir / f"squat_{self.session_id}.csv"
        is_new = not self._csv_path.exists()
        # utf-8-sig: 엑셀에서 한글이 깨지지 않도록
        with open(self._csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["rep", "time", "duration_sec", "min_knee",
                                 "min_hip", "max_torso", "result", "errors"])
            writer.writerow([record.rep, record.time, record.duration_sec,
                             record.min_knee, record.min_hip, record.max_torso,
                             "GOOD" if record.good else "BAD",
                             "|".join(record.errors)])

    def save_summary(self):
        """세션 요약을 JSON으로 저장하고 경로를 반환. 기록이 없으면 저장하지 않음."""
        if not self.save or not self.records:
            return None
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / f"squat_{self.session_id}_summary.json"
        data = self.summary()
        data["reps"] = [asdict(r) for r in self.records]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def print_summary(self):
        s = self.summary()
        ec = s["error_counts"]
        print("\n========== 운동 결과 ==========")
        print(f"총 반복: {s['total_reps']}회 "
              f"(기준 충족 {s['good_reps']}회 / 이탈 {s['bad_reps']}회)")
        print(f"정확도: {s['accuracy']}%")
        for code, (_, desc) in ERROR_INFO.items():
            print(f"  - {desc}: {ec[code]}회")
        if s["total_reps"]:
            print(f"평균 최저 무릎 각도: {s['avg_min_knee']}°")
            print(f"평균 최대 상체 기울기: {s['avg_max_torso']}°")
            print(f"평균 반복 시간: {s['avg_rep_sec']}초")
        print(f"운동 시간: {s['elapsed_sec']}초")
        print("===============================\n")
