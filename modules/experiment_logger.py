"""
experiment_logger.py
--------------------
전처리 조건(OFF / WB / GAMMA / CLAHE / ALL) x 조명 조건별 시행(trial)을 기록한다.

시행 하나가 끝나면 experiment_logs/ 에 두 가지가 저장된다.
  1. trial_<조명>_<조건>_<시각>.csv : 프레임 단위 기록
  2. summary.csv                    : 시행당 한 줄 요약

요약 지표
  detect_rate     : 사람이 검출된 프레임 비율
  valid_rate      : 좌/우 중 한쪽 관절이 모두 visibility 기준을 넘어 각도 계산이 된 비율
  mean_vis        : 검출된 프레임에서 주요 관절 8개의 visibility 평균
  joint_jitter    : 관절 좌표의 2차 차분 크기 (프레임 높이 대비 %)
                    p[t] - 2p[t-1] + p[t-2] 는 부드러운 스쿼트 동작은 상쇄하고
                    프레임마다 튀는 떨림만 남긴다.
  knee_jitter_deg : 무릎 각도의 2차 차분 크기 (도). 좌/우가 바뀐 구간은 제외.
"""

import csv
import time
from pathlib import Path

import numpy as np

MAIN_JOINTS = (11, 12, 23, 24, 25, 26, 27, 28)
JOINT_NAMES = ("l_sh", "r_sh", "l_hip", "r_hip", "l_knee", "r_knee", "l_ank", "r_ank")

FRAME_FIELDS = (
    ["frame_index", "t_sec", "lighting", "condition", "input_mode",
     "detected", "valid", "mean_vis", "side", "knee_angle", "hip_angle",
     "torso_tilt", "phase", "infer_ms", "fps", "pre_ms"]
    + [f"{n}_{axis}" for n in JOINT_NAMES for axis in ("x", "y")]
)

SUMMARY_FIELDS = [
    "trial_id", "lighting", "condition", "input_mode", "n_frames", "duration_s",
    "detect_rate", "valid_rate", "mean_vis", "joint_jitter_pct",
    "knee_jitter_deg", "mean_infer_ms", "mean_fps",
]


def _nanmean(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


class ExperimentLogger:
    def __init__(self, out_dir="experiment_logs", quiet=False):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.quiet = quiet
        self.recording = False
        self.rows = []
        self.positions = []
        self.lighting = ""
        self.condition = ""
        self.trial_id = ""
        self.t0 = 0.0
        self.frame_h = 1

    @property
    def n_frames(self):
        return len(self.rows)

    def start(self, condition, lighting, trial_id=None):
        self.recording = True
        self.rows = []
        self.positions = []
        self.condition = condition
        self.lighting = lighting
        self.trial_id = trial_id or time.strftime("%Y%m%d_%H%M%S")
        self.t0 = None
        if not self.quiet:
            print(f"[REC] start: light={lighting}, cond={condition}")

    def add(self, data, analysis):
        """프레임마다 호출. 녹화 중이 아니면 아무것도 하지 않는다."""
        if not self.recording or data["stage"] != "detect":
            return

        if self.t0 is None:
            self.t0 = data["timestamp"]
        self.frame_h = data["frame_size"][1]

        detected = data["num_poses"] > 0
        valid = bool(analysis and analysis.get("valid"))

        mean_vis = float("nan")
        pos = np.full(16, np.nan)
        if detected:
            lm = data["landmarks_px"][0]
            mean_vis = float(np.mean([lm[i][3] for i in MAIN_JOINTS]))
            for k, i in enumerate(MAIN_JOINTS):
                pos[2 * k], pos[2 * k + 1] = lm[i][0], lm[i][1]

        row = {
            "frame_index": data["frame_index"],
            "t_sec": round(data["timestamp"] - self.t0, 4),
            "lighting": self.lighting,
            "condition": self.condition,
            "input_mode": data["input_mode"],
            "detected": int(detected),
            "valid": int(valid),
            "mean_vis": mean_vis,
            "side": analysis["side"] if valid else "",
            "knee_angle": analysis["knee_angle"] if valid else float("nan"),
            "hip_angle": analysis["hip_angle"] if valid else float("nan"),
            "torso_tilt": analysis["torso_tilt"] if valid else float("nan"),
            "phase": analysis["phase"] if valid else "",
            "infer_ms": data["inference_ms"],
            "fps": data["fps"],
            "pre_ms": data.get("pre_ms", float("nan")),
        }
        for k, name in enumerate(JOINT_NAMES):
            row[f"{name}_x"] = pos[2 * k]
            row[f"{name}_y"] = pos[2 * k + 1]

        self.rows.append(row)
        self.positions.append(pos)

    def stop(self, write_summary=True):
        """녹화를 끝내고 CSV를 저장한다. 요약 dict를 반환한다.
        write_summary=False이면 summary.csv에는 추가하지 않는다. (배치 평가용)"""
        self.recording = False
        if len(self.rows) < 3:
            print("[REC] too few frames, discarded.")
            return None

        base = f"trial_{self.lighting}_{self.condition}_{self.trial_id}"
        with open(self.dir / f"{base}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FRAME_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)

        summary = self._summarize()
        if write_summary:
            summary_path = self.dir / "summary.csv"
            is_new = not summary_path.exists()
            with open(summary_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
                if is_new:
                    writer.writeheader()
                writer.writerow(summary)

        if not self.quiet:
            print("[REC] saved:", base)
            print("      " + ", ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in summary.items() if k not in ("trial_id",)))
        return summary

    def _summarize(self):
        rows = self.rows
        n = len(rows)
        detected = np.array([r["detected"] for r in rows], dtype=bool)
        valid = np.array([r["valid"] for r in rows], dtype=bool)

        # 관절 좌표 2차 차분 (연속 3프레임이 모두 검출된 구간만)
        pos = np.array(self.positions).reshape(n, 8, 2)
        d2 = pos[2:] - 2 * pos[1:-1] + pos[:-2]
        mag = np.linalg.norm(d2, axis=2)                       # (n-2, 8)
        ok = np.isfinite(mag).all(axis=1)
        joint_jitter = (float(mag[ok].mean() / self.frame_h * 100.0)
                        if ok.any() else float("nan"))

        # 무릎 각도 2차 차분 (좌/우가 같은 연속 3프레임만)
        knee = np.array([r["knee_angle"] for r in rows], dtype=float)
        sides = [r["side"] for r in rows]
        k2 = np.abs(knee[2:] - 2 * knee[1:-1] + knee[:-2])
        same = np.array([sides[i] != "" and sides[i] == sides[i + 1] == sides[i + 2]
                         for i in range(n - 2)], dtype=bool)
        k2 = k2[same]
        knee_jitter = _nanmean(k2)

        modes = sorted({str(r["input_mode"]) for r in rows})
        return {
            "trial_id": self.trial_id,
            "lighting": self.lighting,
            "condition": self.condition,
            "input_mode": "/".join(modes),
            "n_frames": n,
            "duration_s": rows[-1]["t_sec"],
            "detect_rate": float(detected.mean()),
            "valid_rate": float(valid.mean()),
            "mean_vis": _nanmean([r["mean_vis"] for r in rows]),
            "joint_jitter_pct": joint_jitter,
            "knee_jitter_deg": knee_jitter,
            "mean_infer_ms": _nanmean([r["infer_ms"] for r in rows if r["detected"]]),
            "mean_fps": _nanmean([r["fps"] for r in rows]),
        }