import cv2
import numpy as np
import time
import os
from ultralytics import YOLO

# --- Configuration ---
model = YOLO("yolov8n-seg.pt")
FOCAL_LENGTH = 450
PERSON_REAL_WIDTH = 50
radar_size = 150
radar_radius = radar_size // 2
radar_center = (radar_radius, radar_size)
THREAT_THRESHOLDS = {"low": 3, "medium": 1.5, "high": 0}
THREAT_COLORS = {"low": (255, 0, 0), "medium": (0, 255, 255), "high": (0, 0, 255)}

toggle_yolo = True
person_db = {}
saved_ids = set()
person_id_counter = 0
min_contour_area = 2000
kernel = np.ones((5, 5), np.uint8)

# --- Helper Functions ---
def get_distance(width_px):
    return (PERSON_REAL_WIDTH * FOCAL_LENGTH) / width_px / 100 if width_px else 0

def assign_id(cx, cy):
    global person_id_counter
    for pid, (px, py) in person_db.items():
        if abs(cx - px) < 60 and abs(cy - py) < 60:
            person_db[pid] = (cx, cy)
            return pid
    person_id_counter += 1
    person_db[person_id_counter] = (cx, cy)
    return person_id_counter

def get_threat_level(d):
    if d > THREAT_THRESHOLDS["low"]:
        return "low"
    elif d > THREAT_THRESHOLDS["medium"]:
        return "medium"
    return "high"

def map_x_to_angle(x, w):
    return (x / w) * 180

def draw_radar(frame, detections):
    h, w = frame.shape[:2]
    radar = np.zeros((radar_size, radar_size, 4), dtype=np.uint8)
    cv2.ellipse(radar, radar_center, (radar_radius, radar_radius), 0, 0, 180, (0, 255, 0, 150), 2)

    for angle in range(0, 181, 30):
        rad = np.radians(angle)
        x = int(radar_center[0] + radar_radius * np.cos(rad))
        y = int(radar_center[1] - radar_radius * np.sin(rad))
        cv2.line(radar, radar_center, (x, y), (0, 255, 0, 150), 1)
        cv2.putText(radar, f"{angle}", (x - 10, y - 5), cv2.FONT_HERSHEY_PLAIN, 0.6, (0, 255, 0, 150), 1)

    for (cx, dm, pid) in detections:
        if dm > 3:
            continue
        ang = np.radians(map_x_to_angle(cx, w))
        scale = min(1, dm / 3)
        x = int(radar_center[0] + scale * radar_radius * np.cos(ang))
        y = int(radar_center[1] - scale * radar_radius * np.sin(ang))
        col = THREAT_COLORS[get_threat_level(dm)]
        cv2.circle(radar, (x, y), 5, (*col, 255), -1)
        cv2.putText(radar, f"{dm:.2f}m", (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255, 200), 1)

    y_off = h - radar_size
    overlay = frame[y_off:y_off + radar_size, 0:radar_size]
    if overlay.shape[:2] == radar.shape[:2]:
        mask = radar[:, :, 3:] / 255.0
        for c in range(3):
            overlay[:, :, c] = (1 - mask[:, :, 0]) * overlay[:, :, c] + mask[:, :, 0] * radar[:, :, c]
        frame[y_off:y_off + radar_size, 0:radar_size] = overlay
    return frame

def process_persons(frame):
    results = model(frame, verbose=False)
    detections = []

    for r in results:
        if not hasattr(r, 'boxes') or r.boxes is None:
            continue

        masks = r.masks.data.cpu().numpy() if r.masks else []

        for i, box in enumerate(r.boxes):
            cls_id = int(box.cls)
            if cls_id != 0 or box.conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            width_px = x2 - x1
            distance_m = get_distance(width_px)
            pid = assign_id(cx, cy)

            if pid not in saved_ids:
                saved_ids.add(pid)
                os.makedirs("persons", exist_ok=True)
                person_img = frame[y1:y2, x1:x2]
                if person_img.size > 0:
                    cv2.imwrite(f"persons/person{pid}.jpg", person_img)

            label = f"ID:{pid} {distance_m:.2f}m"
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 15
            cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            if r.masks is not None and i < len(masks):
                mask = (masks[i] * 255).astype(np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                if mask.shape != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) > min_contour_area:
                        cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            detections.append((cx, distance_m, pid))

    return frame, detections

# ---------- MAIN ----------
photo_path = "D:/radar/aaswat.jpeg"  # ✅ Correct full path to your image
frame = cv2.imread(photo_path)
if frame is None:
    print("[ERROR] Image not found at", photo_path)
    exit()

frame = cv2.resize(frame, (1280, 720))  # Optional: Resize to standard
frame, detections = process_persons(frame)
frame = draw_radar(frame, detections)

cv2.putText(frame, f"Static Image Mode", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow("Radar Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
