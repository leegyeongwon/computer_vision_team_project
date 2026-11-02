import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Squat analysis module (angle calculation, posture feedback, overlay drawing)
from squat_analyzer import process_squat


POSE_WINDOW = "MediaPipe Pose Detection"
MASK_WINDOW = "MOG2 Foreground Mask"

# Set landmark connections and the indices of joints to highlight
connections = vision.PoseLandmarksConnections.POSE_LANDMARKS
main_joints = (11, 12, 23, 24, 25, 26, 27, 28)
visibility_threshold = 0.5


def create_subtractor():
    # Create MOG2 to learn the background and generate foreground masks
    return cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=False
    )


def main():
    # Load the model from this folder and configure detection for one person
    model_path = str(Path(__file__).parent / "pose_landmarker_lite.task")
    base_option = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(base_options=base_option, num_poses=1)
    pose_detector = vision.PoseLandmarker.create_from_options(options)

    # Open the default camera
    cap = cv2.VideoCapture(0)

    # Release resources and exit if the camera cannot be opened
    if not cap.isOpened():
        pose_detector.close()
        cap.release()
        raise SystemExit("Cannot open webcam. Check camera permissions and connection.")

    background_subtractor = create_subtractor()

    # Track whether the background is fixed and MOG2 input is enabled
    background_ready = False
    use_mog2 = False
    # Track which windows are currently open
    pose_window_open = False
    mask_window_open = False

    # Create a 5x5 elliptical kernel to fill small holes in the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Track the time of the previous frame and a smoothed FPS value
    prev_time = time.perf_counter()
    fps = 0.0
    frame_index = 0

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

        # ------------------------------------------------------------
        # Setup stage: show only the MOG2 mask window
        # ------------------------------------------------------------
        if not background_ready:
            # Close the MediaPipe window if it is still open (e.g. after reset)
            if pose_window_open:
                cv2.destroyWindow(POSE_WINDOW)
                pose_window_open = False

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
            # Use the full original frame as the default input region
            x1, y1, x2, y2 = 0, 0, w, h
            input_frame = frame.copy()
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

                    # Crop the ROI while preserving the original pixels
                    input_frame = frame[y1:y2, x1:x2].copy()
                    input_mode = "MOG2 ROI"
                # Keep the full original frame if no valid region is found
                else:
                    input_mode = "Original (no ROI)"

            roi = (x1, y1, x2, y2)

            # Get the input dimensions and convert the image to RGB for MediaPipe
            input_h, input_w = input_frame.shape[:2]
            rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

            # Estimate body landmarks from the input image and time the inference
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            t0 = time.perf_counter()
            result = pose_detector.detect(mp_image)
            inference_ms = (time.perf_counter() - t0) * 1000.0

            # Draw the analyzed ROI as a yellow rectangle on the original frame
            if input_mode == "MOG2 ROI":
                cv2.rectangle(
                    frame, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 2
                )

            # Show the input mode and detected pose count at the top
            cv2.putText(frame,
                        f"Input: {input_mode} | Poses: {len(result.pose_landmarks)}",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

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

            # Show the background state and keyboard controls at the bottom
            cv2.putText(frame,
                        "BG: Fixed | m: mode | r: reset | q: quit",
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1)

            # Show FPS at the top right of the pose window
            cv2.putText(frame, f"FPS: {fps:.1f}",
                        (w - 140, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # The pose window itself is shown after handle_frame_data() below,
            # so that anything drawn on annotated_frame there appears on screen
            pose_window_open = True

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

        # b: Freeze the learned background model and open the MediaPipe window
        if key == ord("b"):
            if not background_ready:
                background_ready = True
                print("Background fixed. You can enter the camera view.")

        # m: Switch between original and MOG2 ROI input
        elif key == ord("m"):
            # Allow mode switching only after the background is fixed
            if background_ready:
                use_mog2 = not use_mog2
            # Print instructions if the background is not fixed yet
            else:
                print("Learn the empty background and press b first.")

        # r: Reset the background model, close the MediaPipe window, return to setup
        elif key == ord("r"):
            background_subtractor = create_subtractor()
            background_ready = False
            use_mog2 = False
            if pose_window_open:
                cv2.destroyWindow(POSE_WINDOW)
                pose_window_open = False
            print("Background reset. Leave the camera view, then press b when ready.")

        # q: Exit the loop and continue to resource cleanup below
        elif key == ord("q"):
            break

    # Release the detector and camera, then close all display windows
    pose_detector.close()
    cap.release()
    cv2.destroyAllWindows()


def handle_frame_data(data):
    """Called once per frame, right after processing and before the pose window is shown.

    `data` holds everything computed for the frame (see frame_data in main()).
    Put your own logic here or call your other functions/scripts from here.
    The arrays are reused by the display, so call .copy() before storing them.
    Anything drawn on data["annotated_frame"] here appears in the pose window.
    """
    # Squat posture analysis: draws angles, state and feedback on annotated_frame
    process_squat(data)


if __name__ == "__main__":
    main()