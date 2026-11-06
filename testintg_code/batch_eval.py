"""
batch_eval.py
-------------
녹화한 영상(record.py)을 전처리 조건별로 모두 돌려 CSV로 저장한다.

사용법
    python batch_eval.py recordings/dark_*.avi

    python batch_eval.py recordings/*.avi --roi
    python batch_eval.py recordings/bright_*.avi --degrade none dark

옵션
    --conditions : 돌릴 전처리 조건 (기본: OFF WB GAMMA CLAHE ALL)
    --roi        : 원본 입력과 MOG2 ROI 입력을 둘 다 평가 (기본은 원본 입력만)
    --degrade    : 합성 열화. none/dark (dark = 밝은 영상을 어둡게 + 노이즈)
                   깨끗한 영상(none) + 전처리 OFF의 랜드마크를 기준으로 삼아,
                   열화된 영상에서 전처리 조건별 오차를 ref_* 열에 기록한다.
                   (기준은 모델 자신의 깨끗한 영상 출력이며 사람이 찍은 정답이 아니다.)
    --out        : 출력 폴더 (기본: batch_results)

실시간 파이프라인(main.py)과 같은 방식으로 처리한다.
  - 영상 앞의 빈 배경 구간(bg_end_frame)으로 MOG2 학습과 WB/감마 보정
    (보정은 배경 마지막 15프레임의 평균 사용)
  - 이후 프레임부터 MediaPipe VIDEO 모드로 추론, 녹화 때의 타임스탬프를 그대로 사용
  - MOG2는 전처리 전 프레임(열화는 적용된 프레임)으로 동작

출력
    batch_results/summary_<시각>.csv : 영상 x 열화 x 모드 x 조건당 한 줄
    batch_results/trials/*.csv       : 프레임 단위 기록
"""

import argparse
import csv
import glob
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import sys
sys.path.append(str(Path(__file__).resolve().parent))

from modules.experiment_logger import ExperimentLogger
from modules.preprocess import Preprocessor
from modules.squat_analyzer import analyze_squat

# blur=True, wb=True, gamma=True, clahe=True
PRESETS = {
    "OFF": (False, False, False, False),
    "WB": (False, True, False, False),
    "GAMMA": (False, False, True, False),
    "CLAHE": (False, False, False, True),
    "ALL": (False, True, True, True),
    "ALL + Gausian": (True, True, True, True)
}
DEGRADES = ["none", "dark"]

SUMMARY_FIELDS = [
    "video", "lighting", "degrade", "condition", "input_mode", "n_frames",
    "duration_s", "detect_rate", "valid_rate", "mean_vis", "joint_jitter_pct",
    "knee_jitter_deg", "mean_infer_ms", "mean_pre_ms",
    "ref_joint_err_pct", "ref_knee_err_deg",
]

# ---------------------------------------------------------
# 합성 열화 (프레임 번호만으로 결정되어 재현 가능)
# ---------------------------------------------------------

def degrade(img, kind, idx):
    """none: 그대로, dark: 어둡게 + 센서 노이즈 (CLAHE의 노이즈 증폭까지 평가하기 위함)"""
    if kind == "none":
        return img
    f = img.astype(np.float32) / 255.0
    f = 0.6 * f ** 2.0
    noise = np.random.default_rng(1000 + idx).normal(0, 0.015, f.shape)
    f = f + noise.astype(np.float32)
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------
# MOG2 ROI (main2_exp.py와 같은 로직)
# ---------------------------------------------------------

def mog_roi(mask, w, h):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > w * h * 0.001]
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[0] + b[2] for b in boxes)
    y2 = max(b[1] + b[3] for b in boxes)
    mx, my = int((x2 - x1) * 0.2), int((y2 - y1) * 0.2)
    return max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)


# ---------------------------------------------------------
# 한 번의 실행: 영상 1개 x 입력 모드 x 열화 x 전처리 조건
# ---------------------------------------------------------

def run_one(video, meta, cond, input_mode, kind, model_path, trial_dir):
    flags = PRESETS[cond]
    ts = meta["timestamps_ms"]
    bg_end = int(meta["bg_end_frame"])
    nominal_fps = float(meta.get("fps_measured", 30.0))

    detector = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
    )
    mog = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pre = Preprocessor()

    logger = ExperimentLogger(out_dir=trial_dir, quiet=True)
    logger.start(cond, meta.get("lighting", "unknown"),
                 trial_id=f"{video.stem}_{kind}_{input_mode}")

    cap = cv2.VideoCapture(str(video))
    bg_tail = []
    last_ms = -1
    pre_times = []
    h = 1
    i = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i >= len(ts):
            break

        frame = degrade(frame, kind, i)
        h, w = frame.shape[:2]

        # 빈 배경 구간: MOG2 학습 + 마지막에 WB/감마 보정
        if i < bg_end:
            mog.apply(frame, learningRate=-1)
            if i >= bg_end - 15:
                bg_tail.append(frame.astype(np.float32))
            if i == bg_end - 1:
                pre.calibrate(np.mean(bg_tail, axis=0).astype(np.uint8), verbose=False)
            continue

        mask = mog.apply(frame, learningRate=0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        pre_ms = 0.0
        if any(flags):
            t = time.perf_counter()
            source = pre.apply(frame, *flags)
            pre_ms = (time.perf_counter() - t) * 1000.0
        else:
            source = frame
        pre_times.append(pre_ms)

        x1, y1, x2, y2 = 0, 0, w, h
        label = "Original"
        if input_mode == "roi":
            box = mog_roi(mask, w, h)
            if box:
                x1, y1, x2, y2 = box
                label = "MOG2 ROI"
            else:
                label = "Original (no ROI)"

        input_frame = source[y1:y2, x1:x2]
        ih, iw = input_frame.shape[:2]
        rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ms = max(int(ts[i]), last_ms + 1)
        last_ms = ms

        t = time.perf_counter()
        result = detector.detect_for_video(mp_image, ms)
        infer_ms = (time.perf_counter() - t) * 1000.0

        landmarks = []
        if result.pose_landmarks:
            landmarks = [(int(p.x * iw) + x1, int(p.y * ih) + y1, p.z, p.visibility)
                         for p in result.pose_landmarks[0]]
        analysis = analyze_squat(landmarks) if landmarks else None

        logger.add({
            "stage": "detect",
            "frame_index": i,
            "timestamp": ts[i] / 1000.0,
            "frame_size": (w, h),
            "fps": nominal_fps,
            "inference_ms": infer_ms,
            "pre_ms": pre_ms,
            "input_mode": label,
            "num_poses": 1 if landmarks else 0,
            "landmarks_px": [landmarks] if landmarks else [],
        }, analysis)

    cap.release()
    detector.close()
    summary = logger.stop(write_summary=False)
    if summary is None:
        return None

    return {
        "summary": summary,
        "positions": np.array(logger.positions),
        "knee": np.array([r["knee_angle"] for r in logger.rows], dtype=float),
        "pre_ms": float(np.mean(pre_times)) if pre_times else float("nan"),
        "frame_h": h,
    }


def ref_errors(run, ref):
    """기준(깨끗한 영상 + OFF) 대비 관절 위치 오차(프레임 높이 대비 %)와 무릎각 오차(도)."""
    n = min(len(run["positions"]), len(ref["positions"]))
    p = run["positions"][:n].reshape(n, 8, 2)
    q = ref["positions"][:n].reshape(n, 8, 2)
    err = np.linalg.norm(p - q, axis=2)
    ok = np.isfinite(err).all(axis=1)
    joint = float(err[ok].mean() / run["frame_h"] * 100.0) if ok.any() else float("nan")

    k = np.abs(run["knee"][:n] - ref["knee"][:n])
    k = k[np.isfinite(k)]
    knee = float(k.mean()) if k.size else float("nan")
    return joint, knee


# ---------------------------------------------------------
# 메인
# ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", help="영상 경로 또는 글롭 패턴 (예: recordings/dark_*.avi)")
    ap.add_argument("--conditions", nargs="+", default=list(PRESETS), choices=list(PRESETS))
    ap.add_argument("--degrade", nargs="+", default=["none"], choices=DEGRADES)
    ap.add_argument("--roi", action="store_true", help="MOG2 ROI 입력도 함께 평가")
    ap.add_argument("--out", default="batch_results")
    ap.add_argument("--model", default=str(Path(__file__).parent / "pose_landmarker_lite.task"))
    args = ap.parse_args()

    paths = []
    for pattern in args.videos:
        matched = sorted(glob.glob(pattern))
        paths.extend(Path(m) for m in (matched or [pattern]))

    out = Path(args.out)
    trial_dir = out / "trials"
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / f"summary_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=SUMMARY_FIELDS).writeheader()

    modes = ["orig", "roi"] if args.roi else ["orig"]
    kinds = sorted(args.degrade, key=lambda k: k != "none")   # none을 먼저

    for video in paths:
        meta_path = video.with_suffix(".json")
        if not video.exists() or not meta_path.exists():
            print(f"SKIP {video}: video or .json not found")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not meta.get("bg_end_frame"):
            print(f"SKIP {video}: no empty-background segment")
            continue

        cache = {}

        def get(mode, kind, cond):
            key = (mode, kind, cond)
            if key not in cache:
                print(f"  run {video.stem} | {mode} | {kind} | {cond}", flush=True)
                cache[key] = run_one(video, meta, cond, mode, kind, args.model, trial_dir)
            return cache[key]

        for mode in modes:
            for kind in kinds:
                print(f"\n== {video.name} | light={meta.get('lighting')} | degrade={kind} | input={mode} ==")
                rows = []
                for cond in args.conditions:
                    res = get(mode, kind, cond)
                    if res is None:
                        print(f"  {cond}: too few frames, skipped")
                        continue

                    row = dict(res["summary"])
                    row.update(video=video.stem, degrade=kind, mean_pre_ms=res["pre_ms"],
                               ref_joint_err_pct=float("nan"), ref_knee_err_deg=float("nan"))
                    if kind != "none":
                        ref = get(mode, "none", "OFF")
                        if ref is not None:
                            row["ref_joint_err_pct"], row["ref_knee_err_deg"] = ref_errors(res, ref)
                    rows.append(row)

                    with open(summary_path, "a", newline="", encoding="utf-8") as f:
                        csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore").writerow(row)

                print(f"{'cond':6} {'detect':>7} {'valid':>7} {'vis':>6} {'jit%':>6} "
                      f"{'kneeJ':>6} {'refJ%':>6} {'refK':>6} {'pre ms':>7}")
                for r in rows:
                    print(f"{r['condition']:6} {r['detect_rate']:7.2f} {r['valid_rate']:7.2f} "
                          f"{r['mean_vis']:6.2f} {r['joint_jitter_pct']:6.3f} {r['knee_jitter_deg']:6.2f} "
                          f"{r['ref_joint_err_pct']:6.3f} {r['ref_knee_err_deg']:6.2f} {r['mean_pre_ms']:7.2f}")

    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()