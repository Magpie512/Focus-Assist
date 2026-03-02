import cv2
import mediapipe as mp
import importlib
import os
import sys
import time

# 1. Setup MediaPipe (supports both legacy Solutions API and newer Tasks API)
face_mesh = None
face_landmarker = None

if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
else:
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

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)

video_full_path = os.path.abspath(video_path)

try:
    vlc = importlib.import_module("vlc")
    vlc_instance = vlc.Instance()
    media = vlc_instance.media_new(video_full_path)
    media.add_option("input-repeat=-1")
    alert_player = vlc_instance.media_player_new()
    alert_player.set_media(media)
except Exception as exc:
    print(
        "Error: VLC setup failed. Install dependencies with `pip install -r requirements.txt` "
        f"and ensure VLC media player is installed. Details: {exc}"
    )
    cap.release()
    sys.exit(1)

alert_playing = False
alert_disabled_by_user = False
preview_window_name = "Webcam Preview"
cv2.namedWindow(preview_window_name, cv2.WINDOW_NORMAL)

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

    if face_mesh is not None:
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

    # If VLC was manually closed while active, keep it closed for this run.
    if (
        alert_playing
        and looking_away
        and alert_player.get_state() in (vlc.State.Stopped, vlc.State.Error)
    ):
        alert_playing = False
        alert_disabled_by_user = True

    if looking_away:
        if not alert_playing and not alert_disabled_by_user:
            alert_player.play()
            alert_playing = True
    else:
        if alert_playing:
            alert_player.stop()
            alert_playing = False

    # If user clicks the preview close button, exit without reopening it.
    if cv2.getWindowProperty(preview_window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

    # Basic preview so you can see the detection status
    status_color = (0, 0, 255) if looking_away else (0, 255, 0)
    cv2.putText(frame, f"LOOKING AWAY: {looking_away}", (50, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.imshow(preview_window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if alert_playing:
    alert_player.stop()
if face_mesh is not None:
    face_mesh.close()
if face_landmarker is not None:
    face_landmarker.close()
cv2.destroyAllWindows()