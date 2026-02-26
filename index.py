import cv2
import mediapipe as mp
import os

# 1. Setup MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 2. Check for the .mov file
video_path = "1.mp4"
if not os.path.exists(video_path):
    print(f"Error: {video_path} not found in {os.getcwd()}")
    exit()

# Try opening with the MSMF backend for better .mov support on Windows
video_cap = cv2.VideoCapture(video_path, cv2.CAP_MSMF)
cap = cv2.VideoCapture(0)

video_window_name = "GET BACK TO WORK"
window_open = False

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    looking_away = True 

    if results.multi_face_landmarks:
        # Landmark 4 is the tip of the nose
        nose = results.multi_face_landmarks[0].landmark[4]
        
        # Logic: If nose is centered (0.35 to 0.65), you are focused
        if 0.35 < nose.x < 0.65 and 0.35 < nose.y < 0.65:
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
cv2.destroyAllWindows()