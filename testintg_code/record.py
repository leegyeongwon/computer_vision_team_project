"""
record.py
---------
배치 평가용 원본 영상을 녹화한다. (전처리/오버레이 없이 카메라 프레임 그대로 저장)

사용법
    python record.py --lighting bright
    python record.py --lighting dark
    python record.py --lighting backlit --duration 25 --lock-camera (역광에서 녹화 시)

흐름 (s 키 한 번이면 끝까지 자동으로 진행된다)
    s 누름 -> [카운트다운 5초: 카메라 화면 밖으로 이동]
           -> [빈 배경 3초: 이 구간이 MOG2 학습 + WB/감마 보정에 쓰인다]
           -> [GO: 화면에 들어와 스쿼트 수행, 정해진 시간 후 자동 종료]

저장영상 (recordings/)
    <조명>_<시각>.avi  : 무손실(FFV1) 우선, 안 되면 MJPG
    <조명>_<시각>.json : 조명 라벨, 빈 배경 프레임 수(bg_end_frame), 프레임별 타임스탬프

키: s 시작/취소/조기종료, q 종료
"""

import argparse
import json
import time
from pathlib import Path

import cv2


def open_writer(path, fps, size):
    # 무손실 코덱을 먼저 시도하고, 안 되면 MJPG로 대체한다
    for code in ("FFV1", "MJPG"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*code), fps, size)
        if writer.isOpened():
            return writer, code
    return None, None


def put(img, text, y, color=(0, 255, 255), scale=0.7):
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lighting", default="unknown", help="조명 라벨 (bright/dark/backlit/sidelit 등)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", default="recordings")
    ap.add_argument("--delay", type=float, default=5.0, help="시작 전 카운트다운(초)")
    ap.add_argument("--bg-sec", type=float, default=3.0, help="빈 배경 구간(초)")
    ap.add_argument("--duration", type=float, default=20.0, help="사람이 들어온 뒤 녹화 시간(초)")
    ap.add_argument("--lock-camera", action="store_true",
                    help="카메라 자동 노출/화이트밸런스 끄기 시도 (드라이버에 따라 동작하지 않을 수 있음)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Cannot open webcam. Check camera permissions and connection.")

    if args.lock_camera:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # 0.25 = manual (V4L2/DirectShow 계열)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        print("auto exposure =", cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
              "| auto WB =", cap.get(cv2.CAP_PROP_AUTO_WB))
        print("값이 바뀌지 않았다면 카메라 제조사 앱/설정에서 직접 고정하세요.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    nominal_fps = cap.get(cv2.CAP_PROP_FPS)
    if not nominal_fps or nominal_fps < 1:
        nominal_fps = 30.0

    phase = "idle"          # idle -> countdown -> bg -> action
    t_phase = 0.0
    writer = None
    codec = None
    video_path = None
    timestamps = []
    bg_end = None
    t0 = 0.0
    size = None

    def finish(save):
        nonlocal writer, phase, timestamps, bg_end
        writer.release()
        writer = None
        if save and bg_end:
            meta = {
                "video": video_path.name,
                "lighting": args.lighting,
                "bg_end_frame": bg_end,
                "frame_size": list(size),
                "codec": codec,
                "mirrored": True,
                "fps_measured": round((len(timestamps) - 1) / ((timestamps[-1] - timestamps[0]) / 1000.0), 2)
                if len(timestamps) > 1 else nominal_fps,
                "timestamps_ms": timestamps,
            }
            video_path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
            print(f"Saved: {video_path} ({len(timestamps)} frames, bg_end_frame={bg_end})")
        else:
            video_path.unlink(missing_ok=True)
            print("Recording discarded.")
        phase = "idle"
        timestamps = []
        bg_end = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Cannot read webcam frame.")
            break

        # 실시간 파이프라인과 같은 좌우 반전을 적용해 저장한다
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.perf_counter()
        view = frame.copy()

        if phase == "countdown":
            remain = args.delay - (now - t_phase)
            if remain <= 0:
                size = (w, h)
                stem = f"{args.lighting}_{time.strftime('%Y%m%d_%H%M%S')}"
                video_path = out_dir / f"{stem}.avi"
                writer, codec = open_writer(video_path, nominal_fps, size)
                if writer is None:
                    print("Cannot open a video writer (FFV1/MJPG both failed).")
                    phase = "idle"
                else:
                    t0 = now
                    timestamps = []
                    phase = "bg"
                    print("Recording: empty background...")
            else:
                put(view, f"Leave the camera view! Start in {remain:.1f}s", 40, (0, 0, 255))

        if phase in ("bg", "action"):
            writer.write(frame)                       # 그리기 전의 원본 프레임 저장
            timestamps.append(round((now - t0) * 1000.0, 3))
            elapsed = now - t0

            if phase == "bg":
                put(view, f"REC empty background {elapsed:.1f}/{args.bg_sec:.0f}s (stay out)", 40, (0, 0, 255))
                if elapsed >= args.bg_sec:
                    bg_end = len(timestamps)
                    phase = "action"
                    print("\a>>> GO: enter the view and start squatting <<<")
            else:
                left = args.bg_sec + args.duration - elapsed
                put(view, f"REC GO! {left:.1f}s left  (s: stop early)", 40, (0, 255, 0))
                if left <= 0:
                    finish(save=True)

        if phase == "idle":
            put(view, f"Light: {args.lighting} | s: start | q: quit", 40)

        cv2.imshow("Record", view)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            if phase == "idle":
                phase = "countdown"
                t_phase = time.perf_counter()
                print(f"Starting in {args.delay:.0f}s. Move out of the camera view.")
            elif phase == "countdown":
                phase = "idle"
                print("Cancelled.")
            elif phase == "bg":
                finish(save=False)
            elif phase == "action":
                finish(save=True)
        elif key == ord("q"):
            if phase in ("bg", "action"):
                finish(save=(phase == "action"))
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()