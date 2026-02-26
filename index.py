import cv2
import mediapipe as mp
import os
import sys
import time

# 1. Setup MediaPipe (supports both legacy Solutions API and newer Tasks API)
detector_type = None
face_mesh = None
face_landmarker = None

if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
    detector_type = "solutions"
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
else:
    detector_type = "tasks"
    model_path = os.path.join("models", "face_landmarker.task")
    if not os.path.exists(model_path):
        print(f"Error: Face model not found at {model_path}")
        sys.exit(1)

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
    )
    face_landmarker = FaceLandmarker.create_from_options(options)

# 2. Check for the .mov file
video_path = "1.mov"
if not os.path.exists(video_path):
    print(f"Error: {video_path} not found in {os.getcwd()}")
    sys.exit(1)

# Try opening with the MSMF backend for better .mov support on Windows
video_cap = cv2.VideoCapture(video_path, cv2.CAP_MSMF)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)

if not video_cap.isOpened():
    print(f"Error: Could not open {video_path} for playback.")
    sys.exit(1)

video_window_name = "GET BACK TO WORK"
window_open = False

# Focus window tuning (slightly left-biased to match typical webcam placement)
focus_center_x = 0.46
focus_center_y = 0.50
focus_tolerance_x = 0.20
focus_tolerance_y = 0.17

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    landmarks = None

    if detector_type == "solutions":
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
    else:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        detection_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]

    looking_away = True

    if landmarks:
        # Landmark 4 is the tip of the nose
        nose = landmarks[4]

        # Logic: If nose is within the calibrated focus window, you are focused
        if (
            abs(nose.x - focus_center_x) < focus_tolerance_x
            and abs(nose.y - focus_center_y) < focus_tolerance_y
        ):
            looking_away = False

    if looking_away:
        ret, v_frame = video_cap.read()
        
        # If we hit the end of the .mov, loop it
        if not ret:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, v_frame = video_cap.read()

        if ret and v_frame is not None:
            # Resize video if it's too huge (optional)
            # v_frame = cv2.resize(v_frame, (640, 360))

            cv2.imshow(video_window_name, v_frame)
            window_open = True
    else:
        if window_open:
            try:
                cv2.destroyWindow(video_window_name)
                window_open = False
            except:
                pass

    # Basic preview so you can see the detection status
    status_color = (0, 0, 255) if looking_away else (0, 255, 0)
    cv2.putText(frame, f"LOOKING AWAY: {looking_away}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.imshow("Webcam Preview", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_cap.release()
if face_mesh is not None:
    face_mesh.close()
if face_landmarker is not None:
    face_landmarker.close()
cv2.destroyAllWindows()