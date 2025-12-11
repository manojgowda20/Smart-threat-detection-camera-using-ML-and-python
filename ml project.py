import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import threading
from datetime import datetime

# Import winsound for reliable Windows audio
try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False
    print("⚠️ winsound not available (not on Windows). Alarm will be console-only.")


# ----------------------------------------------------
# CONFIGURATION SETTINGS
# ----------------------------------------------------
ALARM_SOUND = "alarm.wav"  # Path to your .wav file (must be .wav format)
RECORD_SECONDS = 6         # Duration to save suspicious clips
OUT_FOLDER = "suspicious_clips"
LOITER_TIME = 8            # Seconds a person stands still = suspicious
FAST_MOVEMENT_THRESHOLD = 45  # Pixel distance change for fast movement
ALARM_COOLDOWN = 1.5       # Seconds between consecutive alarm triggers

# Create folder for clips
if not os.path.exists(OUT_FOLDER):
    os.makedirs(OUT_FOLDER)

# ----------------------------------------------------
# LOAD MODELS
# ----------------------------------------------------
print("Loading YOLO models...")

# Person detection (YOLOv8 nano)
model_person = YOLO("yolov8n.pt")

# Weapon/fight models (OPTIONAL – if file not available, detection is skipped)
weapon_model = None
fight_model = None
try:
    weapon_model = YOLO("weapon_yolo.pt")
    print("Weapon model loaded.")
except:
    print("⚠️ No weapon_yolo.pt found — weapon detection disabled.")

try:
    fight_model = YOLO("fight_yolo.pt")
    print("⚠️ No fight_yolo.pt found — fight detection disabled.")
except:
    pass # Catch any other model loading error

# ----------------------------------------------------
# Alarm Function (FIXED: Uses winsound for reliability)
# ----------------------------------------------------
# ----------------------------------------------------
# Alarm Function (FIXED: Uses winsound.Beep for reliability)
# ----------------------------------------------------
def play_alarm():
    """Plays a system beep (Windows only)."""
    if not WINSOUND_AVAILABLE:
        print("\a") # Unix/Linux/Mac beep fallback
        return
    
    try:
        print("Beep! (Alarm tone)")
        # Play a loud, high-pitched beep 3 times
        for i in range(3):
            winsound.Beep(1500, 500) # Frequency = 1500Hz, Duration = 300ms
            time.sleep(0.1)
    except Exception as e:
        print(f"Alarm failure: {e}")
# ----------------------------------------------------
# Start camera
# ----------------------------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 15)

prev_center = None
loiter_start = None
recording = False
video_writer = None
last_alarm_time = time.time()

print("\nCamera Ready! Press Q or ESC to stop.\n")

# ----------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 720))
    display_frame = frame.copy()

    suspicious = False
    reason = ""
    persons_detected = 0

    # --------------------------------------
    # 1. PERSON DETECTION & MOVEMENT
    # --------------------------------------
    results = model_person(frame, stream=True, verbose=False)

    current_person_center = None
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # person
                persons_detected += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                current_person_center = (cx, cy)

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.circle(display_frame, (cx, cy), 8, (255, 0, 0), -1)

                # Movement Checks
                if prev_center is None:
                    prev_center = current_person_center
                    loiter_start = time.time()

                dist = np.linalg.norm(np.array(current_person_center) - np.array(prev_center))

                # Loitering Check (if person is standing still)
                if dist < 8:
                    if time.time() - loiter_start > LOITER_TIME:
                        suspicious = True
                        reason = "Loitering Detected ⚠️"
                else:
                    loiter_start = time.time()

                # Fast movement detection
                if dist > FAST_MOVEMENT_THRESHOLD:
                    suspicious = True
                    reason = "Fast Movement 🏃💨"

                prev_center = current_person_center
                
                # NOTE: Breaks out of inner loop to process only the first detected person for movement checks

    # --------------------------------------
    # 2. WEAPON DETECTION (optional)
    # --------------------------------------
    if weapon_model is not None:
        w_results = weapon_model(frame, stream=True, verbose=False)
        for r in w_results:
            for box in r.boxes:
                # Assuming custom model detects weapons (class 0 or custom label)
                suspicious = True
                reason = "Weapon Detected 🔫"
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

    # --------------------------------------
    # 3. FIGHT DETECTION (optional)
    # --------------------------------------
    if fight_model is not None:
        f_results = fight_model(frame, stream=True, verbose=False)
        for r in f_results:
            for box in r.boxes:
                # Assuming custom model detects fight (class 0 or custom label)
                suspicious = True
                reason = "Physical Fight Detected 🤼"
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 4)


    # --------------------------------------
    # Show status
    # --------------------------------------
    status_color = (0, 255, 0)

    if suspicious:
        status_color = (0, 0, 255)
    
    status_text = reason if suspicious else "Normal Activity 😊"

    cv2.putText(display_frame, status_text, (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)

    cv2.imshow("AI Smart Camera – Advanced Suspicious Activity Detection", display_frame)

    # --------------------------------------
    # 4. ALARM & VIDEO CLIP LOGIC
    # --------------------------------------
    if suspicious:
        # Play alarm only if enough time passed since last alarm (cooldown)
        if time.time() - last_alarm_time > ALARM_COOLDOWN:
            threading.Thread(target=play_alarm).start()
            last_alarm_time = time.time()

        # Start recording if not already recording
        if not recording:
            print(f"⚠️ Suspicious Activity – Recording Clip: {reason}")
            recording = True

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(OUT_FOLDER, f"susp_{timestamp}.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            video_writer = cv2.VideoWriter(filename, fourcc, 
                                           15, (960, 720))

            clip_end_time = time.time() + RECORD_SECONDS

    # Recording logic (continues even if suspicious=False to finish clip)
    if recording:
        video_writer.write(frame)
        if time.time() > clip_end_time:
            video_writer.release()
            recording = False
            print("Clip Saved ✔")

    # Exit on 'q' or 'ESC'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

# Clean up
cap.release()
if video_writer is not None and video_writer.isOpened():
    video_writer.release()
cv2.destroyAllWindows()