"""
main.py
------------

추가된 기능
  - 전처리(WB -> 감마 -> CLAHE)를 MediaPipe 입력에만 적용 (MOG2는 원본 프레임 유지)
  - WB/감마는 b 키로 배경을 고정하는 순간 한 번만 보정, CLAHE만 매 프레임 적용
  - 조건 전환: OFF / WB / GAMMA / CLAHE / ALL
  - 조명 라벨 전환: bright / dark / backlit / sidelit
  - 시행 녹화 후 CSV 저장 (experiment_logger.py)

키 조작
  b: 배경 고정(전처리 보정 포함)   m: Original <-> MOG2 ROI
  p: 전처리 조건 순환              l: 조명 라벨 순환
  e: 녹화 시작/종료                r: 리셋   q: 종료
"""

import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Squat analysis module (angle calculation, posture feedback, overlay drawing)
from modules.squat_analyzer import process_squat, toggle_panel, new_set, finish_squat_session
from modules.preprocess import Preprocessor
from modules.experiment_logger import ExperimentLogger


POSE_WINDOW = "MediaPipe Pose Detection (VIDEO)"
MASK_WINDOW = "MOG2 Foreground Mask"
INPUT_WINDOW = "MediaPipe Input"

# Set landmark connections and the indices of joints to highlight
connections = vision.PoseLandmarksConnections.POSE_LANDMARKS
main_joints = (11, 12, 23, 24, 25, 26, 27, 28)
visibility_threshold = 0.5


# 전처리 조건: (이름, (blur, wb, gamma, clahe))
PRESETS = [
    ("OFF", (True, False, False, False)),
    ("WB", (True, True, False, False)),
    ("GAMMA", (True, False, True, False)),
    ("CLAHE", (True, False, False, True)),
    ("WB+GAMMA", (True, True, True, False)),   # 추가
    ("ALL", (True, True, True, True)),
]
LIGHTING = ["bright", "dark", "backlit", "sidelit"]

# handle_frame_data()에서 쓰는 전역 로거
logger = ExperimentLogger()


def create_subtractor():
    # Create MOG2 to learn the background and generate foreground masks
    return cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=False
    )


def main():
    # Load the model from this folder and configure detection for one person
    # running_mode=VIDEO: the detector tracks landmarks across consecutive frames
    model_path = str(Path(__file__).parent / "pose_landmarker_lite.task")
    base_option = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_option,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )
    pose_detector = vision.PoseLandmarker.create_from_options(options)

    # Open the default camera
    cap = cv2.VideoCapture(0)

    # Release resources and exit if the camera cannot be opened
    if not cap.isOpened():
        pose_detector.close()
        cap.release()
        raise SystemExit("Cannot open webcam. Check camera permissions and connection.")

    background_subtractor = create_subtractor()

    # Preprocessing and experiment state
    pre = Preprocessor()
    preset_idx = 0
    lighting_idx = 0

    # Track whether the background is fixed and MOG2 input is enabled
    background_ready = False
    use_mog2 = False
    # Track which windows are currently open
    pose_window_open = False
    mask_window_open = False
    input_window_open = False

    # Create a 5x5 elliptical kernel to fill small holes in the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Track the time of the previous frame and a smoothed FPS value
    prev_time = time.perf_counter()
    fps = 0.0
    frame_index = 0

    # VIDEO mode needs strictly increasing timestamps in milliseconds
    start_time = prev_time
    last_timestamp_ms = -1

    # Read, analyze, and display camera frames continuously
    while True:
        ret, frame = cap.read()

        # Stop the loop if a frame cannot be read
        if not ret:
            print("Cannot read webcam frame.")
            break

        # Mirror the frame for the front-facing camera
        # Comment out the next line when using a rear-facing camera
        frame = cv2.flip(frame, 1)

        # Keep a clean copy of the frame before anything is drawn on it
        raw_frame = frame.copy()

        # Measure FPS from the time between frames and smooth it to reduce flicker
        now = time.perf_counter()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            instant_fps = 1.0 / dt
            fps = instant_fps if fps == 0.0 else 0.9 * fps + 0.1 * instant_fps

        # MOG2는 항상 원본 프레임으로 학습/적용한다 (전처리 영향 방지)
        # Extract the foreground without updating the fixed background model
        if background_ready:
            foreground_mask = background_subtractor.apply(frame, learningRate=0)
        # Update the background model each frame until it is fixed
        else:
            foreground_mask = background_subtractor.apply(frame, learningRate=-1)

        # Apply closing to fill small black holes in the foreground mask
        foreground_mask = cv2.morphologyEx(
            foreground_mask, cv2.MORPH_CLOSE, kernel
        )

        h, w = frame.shape[:2]

        # Per-frame outputs; defaults are used when a stage does not fill them
        stage = "detect" if background_ready else "setup"
        roi = (0, 0, w, h)
        input_mode = None
        input_frame = None
        inference_ms = 0.0
        landmarks_px = []
        landmarks_norm = []
        world_landmarks = []
        cond_name, (use_blur, use_wb, use_gamma, use_clahe) = PRESETS[preset_idx]

        # ------------------------------------------------------------
        # Setup stage: show only the MOG2 mask window
        # ------------------------------------------------------------
        if not background_ready:
            # Close the MediaPipe windows if still open (e.g. after reset)
            if pose_window_open:
                cv2.destroyWindow(POSE_WINDOW)
                pose_window_open = False
            if input_window_open:
                cv2.destroyWindow(INPUT_WINDOW)
                input_window_open = False

            # Draw on a copy so the mask returned in the frame data stays clean
            mask_view = foreground_mask.copy()
            cv2.putText(mask_view,
                        "BG: Learning | b: fix | r: reset | q: quit",
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, 255, 1)
            cv2.putText(mask_view,
                        "Leave the camera view, then press b.",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
            # Show FPS at the top right of the mask window
            cv2.putText(mask_view, f"FPS: {fps:.1f}",
                        (w - 140, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
            cv2.imshow(MASK_WINDOW, mask_view)
            mask_window_open = True

        # ------------------------------------------------------------
        # Detection stage: background is fixed, run MediaPipe
        # ------------------------------------------------------------
        else:
            # 전처리는 ROI 크롭보다 앞, 전체 프레임에 적용한다
            x1, y1, x2, y2 = 0, 0, w, h
            if pre.calibrated and (use_blur or use_wb or use_gamma or use_clahe):
                source = pre.apply(frame, blur=use_blur, wb=use_wb, gamma=use_gamma, clahe=use_clahe)
            else:
                source = frame
            input_frame = source.copy()
            input_mode = "Original"

            # Find a foreground region of interest (ROI) when MOG2 is enabled
            if use_mog2:
                contours, _ = cv2.findContours(
                    foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                # Ignore small regions and collect bounding boxes for foreground candidates
                boxes = [cv2.boundingRect(c) for c in contours
                         if cv2.contourArea(c) > w * h * 0.001]

                # Calculate one rectangle enclosing all valid foreground candidates
                if boxes:
                    x1 = min(box[0] for box in boxes)
                    y1 = min(box[1] for box in boxes)
                    x2 = max(box[0] + box[2] for box in boxes)
                    y2 = max(box[1] + box[3] for box in boxes)

                    # Add 20% padding on each side to reduce clipping of hands and feet
                    margin_x = int((x2 - x1) * 0.2)
                    margin_y = int((y2 - y1) * 0.2)

                    # Clamp the rectangle coordinates to the image boundaries
                    x1 = max(0, x1 - margin_x)
                    y1 = max(0, y1 - margin_y)
                    x2 = min(w, x2 + margin_x)
                    y2 = min(h, y2 + margin_y)

                    # Crop the ROI from the (preprocessed) source
                    input_frame = source[y1:y2, x1:x2].copy()
                    input_mode = "MOG2 ROI"
                # Keep the full original frame if no valid region is found
                else:
                    input_mode = "Original (no ROI)"

            roi = (x1, y1, x2, y2)

            # Get the input dimensions and convert the image to RGB for MediaPipe
            input_h, input_w = input_frame.shape[:2]
            rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

            # Estimate body landmarks with tracking (VIDEO mode) and time the inference
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Timestamp in milliseconds since start; must strictly increase every call
            timestamp_ms = int((now - start_time) * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            t0 = time.perf_counter()
            result = pose_detector.detect_for_video(mp_image, timestamp_ms)
            inference_ms = (time.perf_counter() - t0) * 1000.0

            # Draw the analyzed ROI as a yellow rectangle on the original frame
            if input_mode == "MOG2 ROI":
                cv2.rectangle(
                    frame, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 2
                )

            # Show the input mode, preprocessing condition and detected pose count
            cv2.putText(frame,
                        f"Input: {input_mode} | Pre: {cond_name} | Poses: {len(result.pose_landmarks)}",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Draw landmark points and connections for each detected person
            for pose in result.pose_landmarks:
                # Map input landmark coordinates to pixel positions in the original frame
                points = [(int(p.x * input_w) + x1,
                           int(p.y * input_h) + y1) for p in pose]

                # Store landmarks as (x, y, z, visibility) in original-frame pixels
                landmarks_px.append(
                    [(pt[0], pt[1], p.z, p.visibility)
                     for pt, p in zip(points, pose)]
                )
                # Store the same landmarks with x and y normalized to the original frame
                landmarks_norm.append(
                    [(pt[0] / w, pt[1] / h, p.z, p.visibility)
                     for pt, p in zip(points, pose)]
                )

                # Draw lines between connected landmarks
                for c in connections:
                    # Draw a line only when both endpoints meet the visibility threshold
                    if (pose[c.start].visibility >= visibility_threshold
                            and pose[c.end].visibility >= visibility_threshold):
                        cv2.line(frame, points[c.start], points[c.end], (0, 255, 0), 2)

                # Draw a point at each landmark position
                for i, point in enumerate(points):
                    # Skip landmarks with low visibility
                    if pose[i].visibility < visibility_threshold:
                        continue

                    # Use larger red dots for main joints and smaller blue dots for the others
                    color = (0, 0, 255) if i in main_joints else (255, 0, 0)
                    cv2.circle(frame, point, 6 if i in main_joints else 4, color, -1)

            # Store 3D world landmarks (meters, origin at the hip center)
            world_landmarks = [
                [(p.x, p.y, p.z, p.visibility) for p in pose]
                for pose in result.pose_world_landmarks
            ]

            # Show a full-body visibility reminder when no pose is detected
            if not result.pose_landmarks:
                cv2.putText(frame, "No pose detected. Show your full body.", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Show the keyboard controls at the bottom
            cv2.putText(frame,
                        "m:mode p:pre l:light e:rec r:reset q:quit",
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1)

            # Show FPS at the top right of the pose window
            cv2.putText(frame, f"FPS: {fps:.1f}",
                        (w - 140, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # The pose window itself is shown after handle_frame_data() below,
            # so that anything drawn on annotated_frame there appears on screen
            pose_window_open = True

            # Show exactly what MediaPipe receives (for checking the preprocessing)
            cv2.imshow(INPUT_WINDOW, input_frame)
            input_window_open = True

            # Close the MOG2 mask window once the background is fixed
            if mask_window_open:
                cv2.destroyWindow(MASK_WINDOW)
                mask_window_open = False

        # ------------------------------------------------------------
        # Collect everything computed for this frame and pass it on
        # ------------------------------------------------------------
        frame_data = {
            # Frame bookkeeping
            "frame_index": frame_index,
            "timestamp": now,                    # perf_counter seconds
            "frame_size": (w, h),
            "fps": fps,                          # smoothed FPS of the whole loop
            "inference_ms": inference_ms,        # MediaPipe inference time (0 in setup)
            # State
            "stage": stage,                      # "setup" or "detect"
            "background_ready": background_ready,
            "use_mog2": use_mog2,
            "input_mode": input_mode,            # None in setup stage
            # Experiment state
            "condition": cond_name,              # OFF / WB / GAMMA / CLAHE / ALL
            "lighting": LIGHTING[lighting_idx],  # user-selected lighting label
            # Region analyzed by MediaPipe
            "roi": roi,                          # (x1, y1, x2, y2) in original pixels
            # Pose results (one entry per detected person)
            "num_poses": len(landmarks_px),
            "landmarks_px": landmarks_px,        # [pose][idx] = (x_px, y_px, z, visibility)
            "landmarks_norm": landmarks_norm,    # [pose][idx] = (x/w, y/h, z, visibility)
            "world_landmarks": world_landmarks,  # [pose][idx] = (x, y, z, visibility) in meters
            # Foreground information from MOG2
            "foreground_mask": foreground_mask,  # uint8 mask, 255 = foreground
            "foreground_ratio": cv2.countNonZero(foreground_mask) / (w * h),
            # Images
            "frame": raw_frame,                  # mirrored frame without any drawing
            "annotated_frame": frame,            # frame with ROI, landmarks and text
            "input_frame": input_frame,          # exact image fed to MediaPipe (None in setup)
        }
        handle_frame_data(frame_data)
        frame_index += 1

        # Display the frame with landmarks, the ROI, status text and handler overlays
        if stage == "detect":
            cv2.imshow(POSE_WINDOW, frame)

        # Update the windows and read one keyboard event
        key = cv2.waitKey(1) & 0xFF

        # b: Freeze the learned background model and calibrate the preprocessing
        if key == ord("b"):
            if not background_ready:
                background_ready = True
                pre.calibrate(raw_frame)        # 빈 배경으로 WB/감마 보정
                print("Background fixed. You can enter the camera view.")

        # m: Switch between original and MOG2 ROI input
        elif key == ord("m"):
            if not background_ready:
                print("Learn the empty background and press b first.")
            elif logger.recording:
                print("Stop recording (e) before changing the input mode.")
            else:
                use_mog2 = not use_mog2

        # p: Cycle the preprocessing condition (OFF -> WB -> GAMMA -> CLAHE -> ALL)
        elif key == ord("p"):
            if not background_ready:
                print("Learn the empty background and press b first.")
            elif logger.recording:
                print("Stop recording (e) before changing the condition.")
            else:
                preset_idx = (preset_idx + 1) % len(PRESETS)
                print(f"Preprocessing: {PRESETS[preset_idx][0]}")

        # l: Cycle the lighting label used for logging
        elif key == ord("l"):
            if logger.recording:
                print("Stop recording (e) before changing the lighting label.")
            else:
                lighting_idx = (lighting_idx + 1) % len(LIGHTING)
                print(f"Lighting label: {LIGHTING[lighting_idx]}")

        # e: Start / stop recording one trial
        elif key == ord("e"):
            if not background_ready:
                print("Learn the empty background and press b first.")
            elif logger.recording:
                logger.stop()
            else:
                logger.start(PRESETS[preset_idx][0], LIGHTING[lighting_idx])

        # r: Reset the background model, close the MediaPipe window, return to setup
        elif key == ord("r"):
            if logger.recording:
                logger.stop()
            background_subtractor = create_subtractor()
            background_ready = False
            use_mog2 = False
            preset_idx = 0
            pre = Preprocessor()
            if pose_window_open:
                cv2.destroyWindow(POSE_WINDOW)
                pose_window_open = False
            print("Background reset. Leave the camera view, then press b when ready.")

        elif key == ord("s"):
            toggle_panel()
        elif key == ord("x"):
            new_set()

        # q: Exit the loop and continue to resource cleanup below
        elif key == ord("q"):
            break

    # Save an unfinished trial before exiting
    if logger.recording:
        logger.stop()

    # Release the detector and camera, then close all display windows
    pose_detector.close()
    cap.release()
    finish_squat_session()
    cv2.destroyAllWindows()


def handle_frame_data(data):
    """Called once per frame, right after processing and before the pose window is shown.

    `data` holds everything computed for the frame (see frame_data in main()).
    Anything drawn on data["annotated_frame"] here appears in the pose window.
    """
    # Squat posture analysis: draws angles, state and feedback on annotated_frame
    analysis = process_squat(data)

    if data["stage"] != "detect":
        return

    # Record this frame if a trial is running (no-op otherwise)
    logger.add(data, analysis)

    width, height = data["frame_size"]
    annotated = data["annotated_frame"]

    # Show the running mode and MediaPipe inference time
    cv2.putText(annotated,
                f"VIDEO | Infer: {data['inference_ms']:.1f} ms",
                (width - 300, 65), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2)

    # Experiment status line: condition, lighting label, recording state
    if logger.recording:
        status = f"REC {logger.n_frames}f | Light: {data['lighting']} | Pre: {data['condition']}"
        color = (0, 0, 255)
    else:
        status = f"Light: {data['lighting']} | Pre: {data['condition']} | e: record"
        color = (255, 255, 0)
    cv2.putText(annotated, status, (20, height - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


if __name__ == "__main__":
    main()