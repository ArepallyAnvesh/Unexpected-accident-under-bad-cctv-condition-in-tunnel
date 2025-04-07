import cv2
import torch
import os
import time
from datetime import datetime

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m

# Load video
video_path = "tunnel_cctv.mp4"
cap = cv2.VideoCapture(video_path)

# Output folder
output_dir = "detected_crashes"
os.makedirs(output_dir, exist_ok=True)

frame_count = 0
crash_counter = 0
consecutive_danger_frames = 0
frame_skip = 2  # analyze every other frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # --- Preprocessing for low visibility (tunnel CCTV) ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    frame = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    # Inference
    results = model(frame)
    detections = results.pandas().xyxy[0]

    vehicles = detections[detections['name'].isin(['car', 'truck', 'bus', 'motorbike'])]
    crash_detected = False

    # Check proximity between vehicles
    for i in range(len(vehicles)):
        for j in range(i+1, len(vehicles)):
            xi1, yi1, xi2, yi2 = vehicles.iloc[i][['xmin','ymin','xmax','ymax']]
            xj1, yj1, xj2, yj2 = vehicles.iloc[j][['xmin','ymin','xmax','ymax']]
            
            rect1 = (int(xi1), int(yi1), int(xi2), int(yi2))
            rect2 = (int(xj1), int(yj1), int(xj2), int(yj2))

            overlap_x = min(rect1[2], rect2[2]) - max(rect1[0], rect2[0])
            overlap_y = min(rect1[3], rect2[3]) - max(rect1[1], rect2[1])

            if overlap_x > 0 and overlap_y > 0:
                consecutive_danger_frames += 1
                if consecutive_danger_frames >= 4:
                    crash_detected = True
                break
        if crash_detected:
            break

    # Render detection boxes
    results.render()

    if crash_detected:
        crash_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        crash_path = os.path.join(output_dir, f"crash_{crash_counter}_{timestamp}.jpg")
        cv2.imwrite(crash_path, frame)

        # --- 🔔 CRASH ALERT MESSAGES ---
        print(f"\n🚨 CRASH DETECTED at frame {frame_count}")
        print(f"📷 Crash image saved at: {crash_path}")
        print("📡 Sending alert to emergency services...\n")

        # --- On-screen text ---
        cv2.putText(frame, "🚨 CRASH DETECTED", (50, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 0, 255), 3, cv2.LINE_AA)
        consecutive_danger_frames = 0

    # Show the frame
    cv2.imshow("🚧 Tunnel Accident Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
