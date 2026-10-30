from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Load the model from this folder and configure detection for one person
model_path = str(Path(__file__).parent / "pose_landmarker_lite.task")
base_option = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(base_options=base_option, num_poses=1)
pose_detector = vision.PoseLandmarker.create_from_options(options)
# Set landmark connections and the indices of joints to highlight
connections = vision.PoseLandmarksConnections.POSE_LANDMARKS
main_joints = (11, 12, 23, 24, 25, 26, 27, 28)
visibility_threshold = 0.5

# Open the default camera
cap = cv2.VideoCapture(0)

# Release resources and exit if the camera cannot be opened
if not cap.isOpened():
    pose_detector.close()
    cap.release()
    raise SystemExit("Cannot open webcam. Check camera permissions and connection.")

# Create MOG2 to learn the background and generate foreground masks
background_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=False
)

# Track whether the background is fixed and MOG2 input is enabled
background_ready = False
use_mog2 = False

# Create a 5x5 elliptical kernel to fill small holes in the mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

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

    # Use the full original frame as the default input region
    x1, y1, x2, y2 = 0, 0, w, h
    input_frame = frame.copy()
    input_mode = "Original"

    # Find a foreground region of interest (ROI) when MOG2 is enabled and the background is fixed
    if use_mog2 and background_ready:
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

    # Get the input dimensions and convert the image to RGB for MediaPipe
    input_h, input_w = input_frame.shape[:2]
    rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

    # Estimate body landmarks from the input image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = pose_detector.detect(mp_image)

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

    # Show a full-body visibility reminder when no pose is detected
    if not result.pose_landmarks:
        cv2.putText(frame, "No pose detected. Show your full body.", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    # Choose the label for the fixed or learning background state
    bg_status = "Fixed" if background_ready else "Learning"

    # Show the background state and keyboard controls at the bottom
    cv2.putText(frame,
                f"BG: {bg_status} | b: fix | m: mode | r: reset | q: quit",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 255), 1)
    
    # Display the original frame with landmarks, the ROI, and status text
    cv2.imshow("MediaPipe Pose Detection", frame)
    # Display the MOG2 mask with a black background and white foreground
    cv2.imshow("MOG2 Foreground Mask", foreground_mask)
    # Preview the actual MediaPipe input (currently disabled)
    # cv2.imshow("MediaPipe Input", input_frame)

    # Update the windows and read one keyboard event
    key = cv2.waitKey(1) & 0xFF

    # b: Freeze the learned background model
    if key == ord("b"):
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

    # r: Reset the background model and return to original input for relearning
    elif key == ord("r"):
        background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False
        )
        background_ready = False
        use_mog2 = False
        print("Background reset. Leave the camera view, then press b when ready.")

    # q: Exit the loop and continue to resource cleanup below
    elif key == ord("q"):
        break

# Release the detector and camera, then close all display windows
pose_detector.close()
cap.release()
cv2.destroyAllWindows()
