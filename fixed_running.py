# """
# Employee Gesture Tracking System (EMPLOYEE-ONLY + CSV Logging)

# - Detect only employees from images in employees/<Name>/ folders
# - Logs detected employees and gestures into employee_log.csv
# - No employee ID shown
# - Gesture tracking and smoothing remain
# - Graceful exit on Ctrl+C
# """

# import os

# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"

# import cv2
# import time
# import math
# import threading
# import queue
# import csv
# import numpy as np
# import torch
# from datetime import datetime
# from ultralytics import YOLO
# from insightface.app import FaceAnalysis
# import sys

# print("CUDA available:", torch.cuda.is_available())

# # ======================
# # GEOFENCING (DESK ZONE)
# # ======================

# # Define desk area points (clockwise)
# DESK_ZONE = np.array([
#     (200, 150),   # top-left  (more left + up)
#     (1050, 150),  # top-right (more right)
#     (1050, 650),  # bottom-right (more down)
#     (200, 650)    # bottom-left
# ], dtype=np.int32)

# # ======================
# # CRITICAL PARAMETER ZONE
# # ======================

# CRITICAL_ZONE = np.array([
#     (1080, 150),   # top-left  (just outside desk zone)
#     (1350, 150),   # top-right (increased width)
#     (1350, 650),   # bottom-right (same height as desk)
#     (1080, 650)    # bottom-left
# ], dtype=np.int32)

# # ======================
# # VIRTUAL LINE
# # ======================

# LINE_P1 = (500, 100)
# LINE_P2 = (500, 700)




# # ============================================================
# # GLOBAL SETTINGS
# # ============================================================
# FRAME_DOWNSCALE = 640
# FACE_SIZE = 128
# MAX_FPS_LOSS_TOLERANCE = 0.7
# UNKNOWN_THRESHOLD = 0.32

# PHONE_CLASS_ID = 67          # COCO: cell phone
# PHONE_IOU_THRESHOLD = 0.15

# stop_event = threading.Event()
# frame_q = queue.Queue(maxsize=12)
# identity_cache = {}  # track_id -> person data
# next_track_id = 1
# cache_lock = threading.Lock()

# EMP_DIR = "employees"
# EMP_EMBED_CACHE = os.path.join(EMP_DIR, "embeddings_cache")
# os.makedirs(EMP_EMBED_CACHE, exist_ok=True)

# CSV_FILE = "employee_log.csv"
# if not os.path.exists(CSV_FILE):
#     with open(CSV_FILE, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["Timestamp", "Camera", "Employee", "Gesture"])

# SCREENSHOT_DIR = "lazy_screenshots"
# os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# # ============================================================
# # UTILITIES
# # ============================================================

# def crossed_line(p_prev, p_curr, line_x):
#     if p_prev is None:
#         return False
#     return (p_prev[0] < line_x and p_curr[0] >= line_x) or \
#            (p_prev[0] > line_x and p_curr[0] <= line_x)


# def inside_geofence(point, polygon):
#     return cv2.pointPolygonTest(polygon, point, False) >= 0



# def iou(boxA, boxB):
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])

#     inter = max(0, xB - xA) * max(0, yB - yA)
#     areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

#     return inter / (areaA + areaB - inter + 1e-6)


# def near(p1, p2, thresh=45):
#     return np.linalg.norm(p1 - p2) < thresh



# # ============================================================
# # LOAD EMPLOYEE EMBEDDINGS
# # ============================================================
# def load_employee_embeddings():
#     embeddings, names = [], []

#     print("Loading employee embeddings...")
#     face_app = FaceAnalysis(name="buffalo_l")
#     face_app.prepare(
#         ctx_id=0 if torch.cuda.is_available() else -1,
#         det_size=(FACE_SIZE, FACE_SIZE)
#     )

#     for emp_name in os.listdir(EMP_DIR):
#         emp_path = os.path.join(EMP_DIR, emp_name)
#         if not os.path.isdir(emp_path):
#             continue

#         cache_file = os.path.join(EMP_EMBED_CACHE, f"{emp_name}.npy")
#         if os.path.exists(cache_file):
#             embeddings.append(np.load(cache_file))
#             names.append(emp_name)
#             continue

#         for img_file in os.listdir(emp_path):
#             if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
#                 img = cv2.imread(os.path.join(emp_path, img_file))
#                 faces = face_app.get(img)
#                 if faces:
#                     emb = faces[0].embedding
#                     emb /= (np.linalg.norm(emb) + 1e-6)
#                     embeddings.append(emb)
#                     names.append(emp_name)
#                     np.save(cache_file, emb)
#                     print(f"Saved embedding for {emp_name}")
#                     break

#     print(f"Total employees loaded: {len(names)}")
#     return embeddings, names

# EMP_EMBS, EMP_NAMES = load_employee_embeddings()

# def match_employee(embedding):
#     best_score, best_name = -1, "Unknown"
#     for emp_emb, name in zip(EMP_EMBS, EMP_NAMES):
#         score = np.dot(embedding, emp_emb)
#         if score > best_score:
#             best_score, best_name = score, name
#     return best_name if best_score >= UNKNOWN_THRESHOLD else "Unknown"

# # ============================================================
# # TRACKING
# # ============================================================
# def assign_track_id(bbox):
#     global next_track_id
#     x1, y1, x2, y2 = bbox
#     cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

#     with cache_lock:
#         for tid, data in identity_cache.items():
#             bx1, by1, bx2, by2 = data["bbox"]
#             bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
#             if math.dist((cx, cy), (bcx, bcy)) < 120:
#                 data["bbox"] = bbox
#                 data["last_seen"] = time.time()
#                 return tid



#         tid = next_track_id
#         next_track_id += 1
#         identity_cache[tid] = {
#             "name": "Detecting...",
#             "bbox": bbox,
#             "last_seen": time.time(),
#             "last_face_check": 0,
#             "gesture": "Perfect",
#             "prev_gesture": "Perfect",
#             "prev_pts": None,
#             "still_frames": 0,
#             "last_center": None,
#             "crossed": False,
#             "in_critical_zone": False   # ✅ NEW
#         }


#         return tid

# def clean_cache():
#     now = time.time()
#     with cache_lock:
#         for tid in list(identity_cache.keys()):
#             if now - identity_cache[tid]["last_seen"] > MAX_FPS_LOSS_TOLERANCE:
#                 del identity_cache[tid]

# # ============================================================
# # RTSP READER
# # ============================================================
# class RTSPReader(threading.Thread):
#     def __init__(self, url, cam_name):
#         super().__init__(daemon=True)
#         self.url = url
#         self.cam_name = cam_name
#         self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
#         self.running = True
#         self.start()

#     def run(self):
#         while self.running and not stop_event.is_set():
#             ret, frame = self.cap.read()

#             if not ret:
#                 print(f"[WARN] No frame from {self.cam_name}, reconnecting...")
#                 time.sleep(1)
#                 self.cap.release()
#                 time.sleep(1)
#                 self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
#                 continue

#             if not frame_q.full():
#                 frame_q.put((self.cam_name, frame))

#         self.cap.release()

#     def stop(self):
#         self.running = False


# # ============================================================
# # INFERENCE WORKER
# # ============================================================
# class InferenceWorker(threading.Thread):
#     def __init__(self):
#         super().__init__(daemon=True)
#         self.pose_model = YOLO("yolov8n-pose.pt")
#         self.face_app = FaceAnalysis(name="buffalo_l")
#         self.face_app.prepare(
#             ctx_id=0 if torch.cuda.is_available() else -1,
#             det_size=(FACE_SIZE, FACE_SIZE)
#         )
#         self.start()

#     def run(self):
#         while not stop_event.is_set():
#             try:
#                 cam_name, frame = frame_q.get(timeout=1)
#             except queue.Empty:
#                 continue

#             small = cv2.resize(frame, (FRAME_DOWNSCALE, FRAME_DOWNSCALE))
#             pose_results = self.pose_model.predict(
#                 small,
#                 device="cuda" if torch.cuda.is_available() else "cpu",
#                 verbose=False
#             )[0]

            


#             # phone_boxes = [
#             #     list(map(int, box.xyxy[0]))
#             #     for box in results.boxes
#             #     if int(box.cls[0]) == PHONE_CLASS_ID
#             # ]

            




#             for box, kpts in zip(pose_results.boxes, pose_results.keypoints):
#                 if int(box.cls[0]) != 0:
#                     continue

#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 sx, sy = frame.shape[1] / FRAME_DOWNSCALE, frame.shape[0] / FRAME_DOWNSCALE
#                 X1, Y1, X2, Y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)

#                 # Person center point
#                 cx = int((X1 + X2) / 2)
#                 cy = int((Y1 + Y2) / 2)

#                 # ASSIGN TRACK ID FIRST
#                 tid = assign_track_id((X1, Y1, X2, Y2))

#                 # GET PERSON DATA
#                 data = identity_cache[tid]

#                 # NOW SAFE TO USE data
#                 prev_center = data.get("last_center")


#                 if crossed_line(prev_center, (cx, cy), LINE_P1[0]) and not data["crossed"]:
#                     data["crossed"] = True
#                     print(f"[ALERT] {data['name']} crossed virtual line in {cam_name}")

#                     with open(CSV_FILE, "a", newline="") as f:
#                         csv.writer(f).writerow([
#                             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                             cam_name,
#                             data["name"],
#                             "LINE_CROSS"
#                         ])

#                 data["last_center"] = (cx, cy)


#                 # Check geofence
#                 if not inside_geofence((cx, cy), DESK_ZONE):
#                     continue  # IGNORE person outside desk

#                 # ======================
#                 # CRITICAL ZONE CHECK
#                 # ======================

#                 if inside_geofence((cx, cy), CRITICAL_ZONE):
#                     if not data["in_critical_zone"]:
#                         data["in_critical_zone"] = True
#                         print(f"[CRITICAL ALERT] {data['name']} entered critical zone in {cam_name}")

#                         with open(CSV_FILE, "a", newline="") as f:
#                             csv.writer(f).writerow([
#                                 datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                                 cam_name,
#                                 data["name"],
#                                 "CRITICAL_ZONE_ENTRY"
#                             ])
#                 else:
#                     data["in_critical_zone"] = False



#                 data = identity_cache[tid]

#                 crop = frame[Y1:Y2, X1:X2]
#                 if crop.size and (data["name"] == "Detecting..." or time.time() - data["last_face_check"] > 2):
#                     faces = self.face_app.get(crop)
#                     if faces:
#                         emb = faces[0].embedding
#                         emb /= (np.linalg.norm(emb) + 1e-6)
#                         data["name"] = match_employee(emb)
#                         data["last_face_check"] = time.time()

#                 if data["name"] == "Unknown":
#                     continue

#                 pts = kpts.xy[0].cpu().numpy()

#                 nose = pts[0]          # Nose
#                 left_wrist = pts[9]    # Left wrist
#                 right_wrist = pts[10] # Right wrist

#                 final_gesture = "Perfect"

#                 # LAZY = phone near face (hand near nose)
#                 if near(nose, left_wrist) or near(nose, right_wrist):
#                     final_gesture = "Lazy"





                

#                 if data["prev_gesture"] == "Perfect" and final_gesture == "Lazy":
#                     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#                     path = os.path.join(
#                         SCREENSHOT_DIR,
#                         f"{data['name']}_{cam_name}_{ts}.jpg"
#                     )
#                     cv2.imwrite(path, frame)
#                     print("[SCREENSHOT]", path)


#                 if final_gesture != data["gesture"]:
#                     data["gesture"] = final_gesture
#                     with open(CSV_FILE, "a", newline="") as f:
#                         csv.writer(f).writerow([
#                             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                             cam_name,
#                             data["name"],
#                             final_gesture
#                         ])


#                 data["prev_gesture"] = final_gesture
#                 color = (0, 0, 255) if final_gesture == "Lazy" else (0, 255, 0)
#                 cv2.rectangle(frame, (X1, Y1), (X2, Y2), color, 2)
#                 cv2.putText(
#                     frame,
#                     f"{data['name']} ({final_gesture})",
#                     (X1, Y1 - 8),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.8,
#                     color,
#                     2
#                 )

#             clean_cache()
#             cv2.polylines(frame, [DESK_ZONE], True, (255, 0, 0), 2)
#             cv2.polylines(frame, [CRITICAL_ZONE], True, (0, 0, 255), 2)
#             cv2.putText(
#                 frame,
#                 "CRITICAL ZONE",
#                 (CRITICAL_ZONE[0][0], CRITICAL_ZONE[0][1] - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8,
#                 (0, 0, 255),
#                 2
#             )

#             cv2.putText(frame, "DESK ZONE", (DESK_ZONE[0][0], DESK_ZONE[0][1]-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
            
#             cv2.line(frame, LINE_P1, LINE_P2, (0, 0, 255), 2)
#             cv2.putText(frame, "VIRTUAL LINE", (LINE_P1[0]+5, LINE_P1[1]+30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)



#             cv2.imshow(cam_name, frame)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 stop_event.set()

# # ============================================================
# # MAIN
# # ============================================================
# if __name__ == "__main__":
#     CAMERA_URLS = {
#         "Cam3": "rtsp://admin:Admin@25123@192.168.31.200:554/Streaming/Channels/301"}

#     readers = {cam: RTSPReader(url, cam) for cam, url in CAMERA_URLS.items()}
#     worker = InferenceWorker()

#     try:
#         while not stop_event.is_set():
#             time.sleep(0.5)
#     except KeyboardInterrupt:
#         stop_event.set()
#     finally:
#         for r in readers.values():
#             r.stop()
#         cv2.destroyAllWindows()
#         sys.exit(0)


"""
Employee Gesture Tracking System (EMPLOYEE-ONLY + CSV Logging)

- Detect only employees from images in employees/<Name>/ folders
- Logs detected employees and gestures into employee_log.csv
- No employee ID shown
- Gesture tracking and smoothing remain
- Graceful exit on Ctrl+C
"""

import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"

import cv2
import time
import math
import threading
import queue
import csv
import numpy as np
import torch
from datetime import datetime
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import sys

print("CUDA available:", torch.cuda.is_available())

# ======================
# GEOFENCING (DESK ZONE)
# ======================

# Define desk area points (clockwise)
DESK_ZONE = np.array([
    (200, 150),   # top-left  (more left + up)
    (1050, 150),  # top-right (more right)
    (1050, 650),  # bottom-right (more down)
    (200, 650)    # bottom-left
], dtype=np.int32)

# ======================
# CRITICAL PARAMETER ZONE
# ======================

CRITICAL_ZONE = np.array([
    (1080, 150),   # top-left  (just outside desk zone)
    (1350, 150),   # top-right (increased width)
    (1350, 650),   # bottom-right (same height as desk)
    (1080, 650)    # bottom-left
], dtype=np.int32)

# ======================
# VIRTUAL LINE
# ======================

LINE_P1 = (500, 100)
LINE_P2 = (500, 700)




# ============================================================
# GLOBAL SETTINGS
# ============================================================
FRAME_DOWNSCALE = 640
FACE_SIZE = 128
MAX_FPS_LOSS_TOLERANCE = 0.7
UNKNOWN_THRESHOLD = 0.32

PHONE_CLASS_ID = 67          # COCO: cell phone
PHONE_IOU_THRESHOLD = 0.15

MAX_ALLOWED_PEOPLE = 5   # change as needed


stop_event = threading.Event()
frame_q = queue.Queue(maxsize=12)
identity_cache = {}  # track_id -> person data
next_track_id = 1
cache_lock = threading.Lock()

EMP_DIR = "employees"
EMP_EMBED_CACHE = os.path.join(EMP_DIR, "embeddings_cache")
os.makedirs(EMP_EMBED_CACHE, exist_ok=True)

CSV_FILE = "employee_log.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Camera", "Employee", "Gesture"])

SCREENSHOT_DIR = "lazy_screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# ============================================================
# UTILITIES
# ============================================================

def crossed_line(p_prev, p_curr, line_x):
    if p_prev is None:
        return False
    return (p_prev[0] < line_x and p_curr[0] >= line_x) or \
           (p_prev[0] > line_x and p_curr[0] <= line_x)


def inside_geofence(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0



def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter / (areaA + areaB - inter + 1e-6)


def near(p1, p2, thresh=45):
    return np.linalg.norm(p1 - p2) < thresh



# ============================================================
# LOAD EMPLOYEE EMBEDDINGS
# ============================================================
def load_employee_embeddings():
    embeddings, names = [], []

    print("Loading employee embeddings...")
    face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(
        ctx_id=0 if torch.cuda.is_available() else -1,
        det_size=(FACE_SIZE, FACE_SIZE)
    )

    for emp_name in os.listdir(EMP_DIR):
        emp_path = os.path.join(EMP_DIR, emp_name)
        if not os.path.isdir(emp_path):
            continue

        cache_file = os.path.join(EMP_EMBED_CACHE, f"{emp_name}.npy")
        if os.path.exists(cache_file):
            embeddings.append(np.load(cache_file))
            names.append(emp_name)
            continue

        for img_file in os.listdir(emp_path):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                img = cv2.imread(os.path.join(emp_path, img_file))
                faces = face_app.get(img)
                if faces:
                    emb = faces[0].embedding
                    emb /= (np.linalg.norm(emb) + 1e-6)
                    embeddings.append(emb)
                    names.append(emp_name)
                    np.save(cache_file, emb)
                    print(f"Saved embedding for {emp_name}")
                    break

    print(f"Total employees loaded: {len(names)}")
    return embeddings, names

EMP_EMBS, EMP_NAMES = load_employee_embeddings()

def match_employee(embedding):
    best_score, best_name = -1, "Unknown"
    for emp_emb, name in zip(EMP_EMBS, EMP_NAMES):
        score = np.dot(embedding, emp_emb)
        if score > best_score:
            best_score, best_name = score, name
    return best_name if best_score >= UNKNOWN_THRESHOLD else "Unknown"

# ============================================================
# TRACKING
# ============================================================
def assign_track_id(bbox):
    global next_track_id
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    with cache_lock:
        for tid, data in identity_cache.items():
            bx1, by1, bx2, by2 = data["bbox"]
            bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
            if math.dist((cx, cy), (bcx, bcy)) < 120:
                data["bbox"] = bbox
                data["last_seen"] = time.time()
                return tid



        tid = next_track_id
        next_track_id += 1
        identity_cache[tid] = {
            "name": "Detecting...",
            "bbox": bbox,
            "last_seen": time.time(),
            "last_face_check": 0,
            "gesture": "Perfect",
            "prev_gesture": "Perfect",
            "prev_pts": None,
            "still_frames": 0,
            "last_center": None,
            "crossed": False,
            "in_critical_zone": False   # ✅ NEW
        }


        return tid

def clean_cache():
    now = time.time()
    with cache_lock:
        for tid in list(identity_cache.keys()):
            if now - identity_cache[tid]["last_seen"] > MAX_FPS_LOSS_TOLERANCE:
                del identity_cache[tid]

# ============================================================
# RTSP READER
# ============================================================
class RTSPReader(threading.Thread):
    def __init__(self, url, cam_name):
        super().__init__(daemon=True)
        self.url = url
        self.cam_name = cam_name
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.running = True
        self.start()

    def run(self):
        while self.running and not stop_event.is_set():
            ret, frame = self.cap.read()

            if not ret:
                print(f"[WARN] No frame from {self.cam_name}, reconnecting...")
                time.sleep(1)
                self.cap.release()
                time.sleep(1)
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                continue

            if not frame_q.full():
                frame_q.put((self.cam_name, frame))

        self.cap.release()

    def stop(self):
        self.running = False


# ============================================================
# INFERENCE WORKER
# ============================================================
class InferenceWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.pose_model = YOLO("yolov8n-pose.pt")
        self.face_app = FaceAnalysis(name="buffalo_l")
        self.face_app.prepare(
            ctx_id=0 if torch.cuda.is_available() else -1,
            det_size=(FACE_SIZE, FACE_SIZE)
        )
        self.start()

    def run(self):
        while not stop_event.is_set():
            try:
                cam_name, frame = frame_q.get(timeout=1)
            except queue.Empty:
                continue

            small = cv2.resize(frame, (FRAME_DOWNSCALE, FRAME_DOWNSCALE))
            pose_results = self.pose_model.predict(
                small,
                device="cuda" if torch.cuda.is_available() else "cpu",
                verbose=False
            )[0]

            


            # phone_boxes = [
            #     list(map(int, box.xyxy[0]))
            #     for box in results.boxes
            #     if int(box.cls[0]) == PHONE_CLASS_ID
            # ]

            

            current_people_in_desk = set()



            for box, kpts in zip(pose_results.boxes, pose_results.keypoints):
                if int(box.cls[0]) != 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                sx, sy = frame.shape[1] / FRAME_DOWNSCALE, frame.shape[0] / FRAME_DOWNSCALE
                X1, Y1, X2, Y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)

                # Person center point
                cx = int((X1 + X2) / 2)
                cy = int((Y1 + Y2) / 2)

                # ASSIGN TRACK ID FIRST
                tid = assign_track_id((X1, Y1, X2, Y2))

                # GET PERSON DATA
                data = identity_cache[tid]

                # NOW SAFE TO USE data
                prev_center = data.get("last_center")


                if crossed_line(prev_center, (cx, cy), LINE_P1[0]) and not data["crossed"]:
                    data["crossed"] = True
                    print(f"[ALERT] {data['name']} crossed virtual line in {cam_name}")

                    with open(CSV_FILE, "a", newline="") as f:
                        csv.writer(f).writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            cam_name,
                            data["name"],
                            "LINE_CROSS"
                        ])

                data["last_center"] = (cx, cy)


                # Check geofence
                if not inside_geofence((cx, cy), DESK_ZONE):
                    continue  # IGNORE person outside desk

                current_people_in_desk.add(tid)


                # ======================
                # CRITICAL ZONE CHECK
                # ======================

                if inside_geofence((cx, cy), CRITICAL_ZONE):
                    if not data["in_critical_zone"]:
                        data["in_critical_zone"] = True
                        print(f"[CRITICAL ALERT] {data['name']} entered critical zone in {cam_name}")

                        with open(CSV_FILE, "a", newline="") as f:
                            csv.writer(f).writerow([
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                cam_name,
                                data["name"],
                                "CRITICAL_ZONE_ENTRY"
                            ])
                else:
                    data["in_critical_zone"] = False



                data = identity_cache[tid]

                crop = frame[Y1:Y2, X1:X2]
                if crop.size and (data["name"] == "Detecting..." or time.time() - data["last_face_check"] > 2):
                    faces = self.face_app.get(crop)
                    if faces:
                        emb = faces[0].embedding
                        emb /= (np.linalg.norm(emb) + 1e-6)
                        data["name"] = match_employee(emb)
                        data["last_face_check"] = time.time()

                if data["name"] == "Unknown":
                    continue

                pts = kpts.xy[0].cpu().numpy()

                nose = pts[0]          # Nose
                left_wrist = pts[9]    # Left wrist
                right_wrist = pts[10] # Right wrist

                final_gesture = "Perfect"

                # LAZY = phone near face (hand near nose)
                if near(nose, left_wrist) or near(nose, right_wrist):
                    final_gesture = "Lazy"





                

                if data["prev_gesture"] == "Perfect" and final_gesture == "Lazy":
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(
                        SCREENSHOT_DIR,
                        f"{data['name']}_{cam_name}_{ts}.jpg"
                    )
                    cv2.imwrite(path, frame)
                    print("[SCREENSHOT]", path)


                if final_gesture != data["gesture"]:
                    data["gesture"] = final_gesture
                    with open(CSV_FILE, "a", newline="") as f:
                        csv.writer(f).writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            cam_name,
                            data["name"],
                            final_gesture
                        ])


                data["prev_gesture"] = final_gesture
                color = (0, 0, 255) if final_gesture == "Lazy" else (0, 255, 0)
                cv2.rectangle(frame, (X1, Y1), (X2, Y2), color, 2)
                cv2.putText(
                    frame,
                    f"{data['name']} ({final_gesture})",
                    (X1, Y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

            crowd_count = len(current_people_in_desk)


            clean_cache()
            cv2.polylines(frame, [DESK_ZONE], True, (255, 0, 0), 2)
            cv2.polylines(frame, [CRITICAL_ZONE], True, (0, 0, 255), 2)
            cv2.putText(
                frame,
                "CRITICAL ZONE",
                (CRITICAL_ZONE[0][0], CRITICAL_ZONE[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.putText(frame, "DESK ZONE", (DESK_ZONE[0][0], DESK_ZONE[0][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
            
            cv2.line(frame, LINE_P1, LINE_P2, (0, 0, 255), 2)
            cv2.putText(frame, "VIRTUAL LINE", (LINE_P1[0]+5, LINE_P1[1]+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)


            cv2.putText(
                frame,
                f"People in Desk Zone: {crowd_count}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2
            )



            cv2.imshow(cam_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    CAMERA_URLS = {
        "Cam3": "rtsp://admin:Admin@25123@192.168.31.200:554/Streaming/Channels/301"}

    readers = {cam: RTSPReader(url, cam) for cam, url in CAMERA_URLS.items()}
    worker = InferenceWorker()

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        for r in readers.values():
            r.stop()
        cv2.destroyAllWindows()
        sys.exit(0)

