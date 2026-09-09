import os
import re
import csv
import glob
import cv2
import torch
import json
import pickle
import open_clip
import subprocess
import pytesseract
import numpy as np
from PIL import Image
from datetime import datetime, timedelta
from collections import deque, Counter
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
from config import (FRAME_INTERVAL, FRAME_PER_SECOND, PRE_VIOLATION_DURATION,
    POST_VIOLATION_DURATION, ACTIVITIES, ACTIVE_ACTIVITIES, VIOLATIONS, IDLE_ACTIVITIES, TIMESTAMP_CROP)


# ─────────────────────────────────────────────
#  CONFIGURATION (identik dengan inference.py)
# ─────────────────────────────────────────────

INPUT_PATH           = "frames"
OUTPUT_PATH          = "output"

PRE_VIOLATIONS_FRAME = max(1, int(PRE_VIOLATION_DURATION * FRAME_PER_SECOND))
POST_VIOLATION_FRAME = max(1, int(POST_VIOLATION_DURATION * FRAME_PER_SECOND))

SMOOTHING_WINDOW        = 3
VIOLATION_CONFIRM_FRAMES = 5
IOU_THRESHOLD           = 0.15
GRACE_PERIOD_FRAMES     = 30

REID_ENABLED = True
REID_MAX_GAP_FRAMES = 90

REID_MIN_COSINE_SCORE = 0.9
REID_MAX_DISTANCE_PX = 150

# ── Face identification (Fase 3, docs/face-recognition-plan.md) ──
FACE_ID_ENABLED            = True   # master switch, default OFF
FACE_ID_STORE_PATH         = "weights/faces/employees.npz"
FACE_ID_RETRY_EVERY_FRAMES = 15      # jangan coba tiap frame
FACE_ID_MAX_ATTEMPTS       = 8       # setelah ini, person_id ditandai UNKNOWN permanen
FACE_MIN_SCORE             = 0.30
FACE_MIN_MARGIN            = 0.10
FACE_MIN_FACE_PX           = 17

VIDEO_FOURCC = cv2.VideoWriter_fourcc(*'mp4v')

VIDEO_TIMESTAMP_FORMAT = "%d-%m-%Y %H:%M:%S"

DIGIT_TEMPLATES = {}
if os.path.exists("digit_templates.pkl"):
    with open("digit_templates.pkl", "rb") as f:
        DIGIT_TEMPLATES = pickle.load(f)
else:
    print("[WARN] digit_templates.pkl tidak ditemukan, template matching akan selalu fallback ke Tesseract")


# ─────────────────────────────────────────────
#  HELPER: OCR TIMESTAMP (identik dengan inference.py)
# ─────────────────────────────────────────────

def segment_characters(thresh_img, min_w=2, min_h=8):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = [b for b in boxes if b[2] >= min_w and b[3] >= min_h]
    boxes.sort(key=lambda b: b[0])
    return boxes

def match_char(crop, templates, min_confidence=0.5):
    best_char, best_score = None, -1.0
    for ch, tmpl in templates.items():
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            continue
        resized = cv2.resize(crop, (tmpl.shape[1], tmpl.shape[0]))
        res = cv2.matchTemplate(resized, tmpl, cv2.TM_CCOEFF_NORMED)
        score = float(res[0][0])
        if score > best_score:
            best_score, best_char = score, ch
    if best_score < min_confidence:
        return None, best_score
    return best_char, best_score

def extract_cctv_timestamp_template(frame, expected_len=18, min_confidence=0.5):
    xs = [p[0] for p in TIMESTAMP_CROP]
    ys = [p[1] for p in TIMESTAMP_CROP]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.resize(roi_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, roi_thresh = cv2.threshold(roi_gray, 180, 255, cv2.THRESH_BINARY)

    boxes = segment_characters(roi_thresh)

    if len(boxes) != expected_len:
        return None

    result_chars = []
    for (bx, by, bw, bh) in boxes:
        crop = roi_thresh[by:by+bh, bx:bx+bw]
        ch, score = match_char(crop, DIGIT_TEMPLATES, min_confidence)
        if ch is None:
            return None
        result_chars.append(ch)

    raw = "".join(result_chars)
    return f"{raw[:10]} {raw[10:]}"

def extract_cctv_timestamp(frame):
    ts = extract_cctv_timestamp_template(frame)
    if ts:
        return ts

    xs = [p[0] for p in TIMESTAMP_CROP]
    ys = [p[1] for p in TIMESTAMP_CROP]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.resize(roi_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, roi_thresh = cv2.threshold(roi_gray, 180, 255, cv2.THRESH_BINARY)

    text = pytesseract.image_to_string(
        roi_thresh,
        config="--psm 7 -c tessedit_char_whitelist=0123456789-:MonTueWedThuFriSatSun "
    )

    match = re.search(r"(\d{2}-\d{2}-\d{4}).*?(\d{2}:\d{2}:\d{2})", text)
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return None

def compute_frame_timestamp(base_dt, frame_count, fps):
    if base_dt is None:
        return None
    elapsed_seconds = (frame_count - 1) / fps
    ts = base_dt + timedelta(seconds=elapsed_seconds)
    return ts.strftime(VIDEO_TIMESTAMP_FORMAT)

# ZONE_POLYGON = [
#     (462, 1079),
#     (0, 243),
#     (168, 2),
#     (792, 265)
# ]

ZONE_POLYGON_PATH = "zone_polygon.json"

if os.path.exists(ZONE_POLYGON_PATH):
    with open(ZONE_POLYGON_PATH) as f:
        ZONE_POLYGON = [tuple(pt) for pt in json.load(f)]
else:
    # fallback kalau zone_polygon.json belum pernah dibuat lewat define_zone.py
    ZONE_POLYGON = [
        (462, 1079),
        (0, 243),
        (168, 2),
        (792, 265)
    ]

# ─────────────────────────────────────────────
#  HELPER: DRAW LABEL (identik dengan inference.py)
# ─────────────────────────────────────────────

def draw_label(frame, text1, text2, x1, y1, x2, y2, color):
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness  = 2
    padding    = 4

    (w1, h1), _ = cv2.getTextSize(text1, font, font_scale, thickness)
    (w2, h2), _ = cv2.getTextSize(text2, font, font_scale, thickness)

    if y1 - h1 - h2 - padding * 3 >= 0:
        ty2 = y1 - padding
        ty1 = ty2 - h2 - padding
        cv2.putText(frame, text1, (x1, ty1), font, font_scale, color, thickness)
        cv2.putText(frame, text2, (x1, ty2), font, font_scale, color, thickness)
    else:
        cv2.putText(frame, text1, (x1 + padding, y1 + h1 + padding),          font, font_scale, color, thickness)
        cv2.putText(frame, text2, (x1 + padding, y1 + h1 + h2 + padding * 2), font, font_scale, color, thickness)


# ─────────────────────────────────────────────
#  HELPER: Re-ID (identik dengan inference.py)
# ─────────────────────────────────────────────

def cosine_sim(emb_a, emb_b):
    return float(np.dot(emb_a, emb_b))

def euclidean_dist(pos_a, pos_b):
    return float(np.hypot(pos_a[0]-pos_b[0], pos_a[1]-pos_b[1]))


# ─────────────────────────────────────────────
#  VIOLATION VIDEO TRACKER (identik dengan inference.py)
# ─────────────────────────────────────────────

class ViolationVideoTracker:
    def __init__(self, track_id, frame_size, output_dir):
        self.track_id = track_id
        self.frame_size = frame_size
        self.output_dir = output_dir

        self.pre_buffer = deque(maxlen=PRE_VIOLATIONS_FRAME)

        self.state = "IDLE"
        self.writer = None
        self.post_count = 0
        self.clip_index = 0
        self.current_clipname = None

    def _open_writer(self):
        self.clip_index += 1
        filename = f"track{self.track_id:03d}_clip{self.clip_index:02d}.mp4"
        self.current_clipname = os.path.join(self.output_dir, filename)
        w, h = self.frame_size
        self.writer = cv2.VideoWriter(
            self.current_clipname, VIDEO_FOURCC, FRAME_PER_SECOND, (w, h)
        )
        print(f"[VideoTracker] Track {self.track_id}: mulai rekam -> {filename}")

    def _flush_pre_buffer(self):
        for f in self.pre_buffer:
            self.writer.write(f)
        self.pre_buffer.clear()

    def _close_writer(self):
        if self.writer:
            self.writer.release()
            self.writer = None
            print(f"[VideoTracker] Track {self.track_id}: klip selesai -> {self.current_clipname}")

            tmp_path = self.current_clipname.replace(".mp4", "_tmp.mp4")
            os.rename(self.current_clipname, tmp_path)
            subprocess.run([
                "ffmpeg", "-y",
                "-i", tmp_path,
                "-vcodec", "libx264",
                "-crf", "23",
                "-preset", "fast",
                self.current_clipname
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(tmp_path)
            print(f"[VideoTracker] Konversi H.264 selesai -> {self.current_clipname}")

    def push(self, annotated_frame, is_confirmed_violation: bool):
        if self.state == 'IDLE':
            self.pre_buffer.append(annotated_frame.copy())

            if is_confirmed_violation:
                self._open_writer()
                self._flush_pre_buffer()
                self.writer.write(annotated_frame.copy())
                self.state = 'RECORDING'
                self.post_count = 0

        elif self.state == 'RECORDING':
            self.writer.write(annotated_frame.copy())
            self.post_count += 1

            if self.post_count >= POST_VIOLATION_FRAME:
                self._close_writer()
                self.pre_buffer.clear()
                self.state = 'IDLE'

    def finalize(self):
        if self.state in ('RECORDING', 'POST_BUFFER') and self.writer:
            self._close_writer()


# ─────────────────────────────────────────────
#  LOAD MODELS — dilakukan sekali saat import (identik dengan inference.py)
# ─────────────────────────────────────────────

print("Load YOLO model...")
yolo_model = YOLO("yolov8l-worldv2.pt")
yolo_model.set_classes(['person'])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print("Load OpenCLIP model...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k', device=device
)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

activity_tokens = tokenizer(ACTIVITIES).to(device)
with torch.no_grad():
    activity_features = clip_model.encode_text(activity_tokens)
    activity_features /= activity_features.norm(dim=-1, keepdim=True)

print("Load face identifier...")
face_identifier = None
if FACE_ID_ENABLED:
    try:
        from inference.face_identity import FaceIdentifier
        face_identifier = FaceIdentifier(
            store_path=FACE_ID_STORE_PATH,
            min_score=FACE_MIN_SCORE,
            min_margin=FACE_MIN_MARGIN,
            min_face_px=FACE_MIN_FACE_PX,
            device=device,
        )
        print("Load face identifier... OK")
    except Exception as e:
        print(f"[WARN] Face identifier gagal dimuat, identifikasi karyawan dinonaktifkan: {e}")
else:
    print("Load face identifier... dilewati (FACE_ID_ENABLED=False)")

zone = Polygon(ZONE_POLYGON)


# ─────────────────────────────────────────────
#  INFERENCE SESSION — versi streaming dari run_inference()
#
#  3 fase yang dulu digabung dalam 1 fungsi (init di awal, loop
#  per-frame, finalize di akhir) sekarang jadi 3 entry point terpisah
#  yang bisa dipanggil independen di waktu berbeda:
#
#    session = InferenceSession(output_dir)   # panggil 1x, di awal sesi
#    session.process_frame(frame_path)        # panggil berkali-kali, tiap ada frame baru
#    result = session.finalize()              # panggil 1x, setelah frame terakhir selesai
#
#  Semua state yang di run_inference() dulu berupa variabel lokal
#  (track_last_seen, activity_history, dst) sekarang jadi atribut
#  self.* supaya "diingat" antar pemanggilan process_frame().
# ─────────────────────────────────────────────

class InferenceSession:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.frames_dir = os.path.join(output_dir, "frames")
        self.violation_frames_dir = os.path.join(output_dir, "frames/violation")
        self.violation_video_dir = os.path.join(output_dir, "videos/violation")
        self.annotated_video_dir = os.path.join(output_dir, "videos/annotated")

        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.violation_frames_dir, exist_ok=True)
        os.makedirs(self.violation_video_dir, exist_ok=True)
        os.makedirs(self.annotated_video_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.violation_log_path = os.path.join(output_dir, f"violation_log_{timestamp}.csv")
        self.annotated_video_path = os.path.join(self.annotated_video_dir, f"annotated_{timestamp}.mp4")

        # csv_file/writer baru dibuka saat frame pertama BERHASIL dibaca
        # (bukan di __init__), supaya perilakunya sama seperti run_inference()
        # lama: kalau ternyata tidak ada frame sama sekali, tidak ada file
        # violation_log_*.csv yang dibuat.
        self.csv_file = None
        self.writer = None
        self.full_video_writer = None

        self.video_base_dt = None

        # state per track_id (hasil YOLO/bytetrack)
        self.track_last_seen     = {}
        self.activity_history    = {}
        self.violation_counter   = {}
        self.last_smoothed       = {}
        self.last_confirmed      = {}
        self.violation_timestamp = {}

        self.video_trackers: dict[int, ViolationVideoTracker] = {}
        self.person_clip_count = {}

        # state per person_id (hasil re-id)
        self.tid_to_person = {}
        self.person_active_tid = {}
        self.person_last_embedding = {}
        self.person_last_position = {}
        self.person_last_seen_frame = {}
        self.next_person_id = 1

        # state per person_id (hasil face identification, Fase 3)
        self.person_employee = {}          # person_id -> {"employee_id","employee_name","score"} | "UNKNOWN"
        self.person_face_attempts = {}     # person_id -> int
        self.person_last_face_try = {}     # person_id -> frame_count

        self.max_concurrent_persons = 0
        self.frame_count = 0
        self.results_log = []
        self.frame_size = None

    def _resolve_person_id(self, tid, embedding, position, frame_count):
        if REID_ENABLED:
            best_pid, best_sim = None, 0.0
            for pid, last_frame in self.person_last_seen_frame.items():
                if pid in self.person_active_tid:
                    continue
                if frame_count - last_frame > REID_MAX_GAP_FRAMES:
                    continue
                sim = cosine_sim(embedding, self.person_last_embedding[pid])
                dist = euclidean_dist(position, self.person_last_position[pid])
                passed = sim >= REID_MIN_COSINE_SCORE and dist <= REID_MAX_DISTANCE_PX
                print(f"[ReID-DEBUG] frame {frame_count}: track_id {tid} vs person_id {pid} "
                      f"-> sim={sim:.3f} dist={dist:.1f}px {'LOLOS' if passed else 'gagal'}")
                if passed and sim > best_sim:
                    best_sim, best_pid = sim, pid

            if best_pid is not None:
                print(f"[ReID] track_id {tid} dikenali sebagai person_id {best_pid}"
                      f"(cosine_sim:{best_sim:.3f})")
                return best_pid

        pid = self.next_person_id
        self.next_person_id += 1
        return pid

    def _resolve_employee_id(self, person_id, crop, frame_count):
        """Coba identifikasi wajah untuk person_id ini, dengan retry ter-throttle
        dan lock permanen begitu berhasil/menyerah (plan §D5, §6.5). Return dict
        {"employee_id","employee_name","score"} kalau match, None kalau belum/
        tidak berhasil -- TIDAK PERNAH melempar exception ke process_frame()."""
        if face_identifier is None:
            return None

        cached = self.person_employee.get(person_id)
        if cached is not None:
            return cached if cached != "UNKNOWN" else None

        attempts = self.person_face_attempts.get(person_id, 0)
        if attempts >= FACE_ID_MAX_ATTEMPTS:
            self.person_employee[person_id] = "UNKNOWN"
            return None

        last_try = self.person_last_face_try.get(person_id, -FACE_ID_RETRY_EVERY_FRAMES)
        if frame_count - last_try < FACE_ID_RETRY_EVERY_FRAMES:
            return None

        self.person_last_face_try[person_id] = frame_count
        self.person_face_attempts[person_id] = attempts + 1

        try:
            match = face_identifier.identify(crop)
        except Exception as e:
            print(f"[FaceID] identify() error untuk person_id {person_id}: {e}")
            match = None

        if match is None:
            if self.person_face_attempts[person_id] >= FACE_ID_MAX_ATTEMPTS:
                self.person_employee[person_id] = "UNKNOWN"
                print(f"[FaceID] person_id {person_id}: jatah percobaan habis, ditandai UNKNOWN")
            return None

        result = {
            "employee_id": match.employee_id,
            "employee_name": match.employee_name,
            "score": match.score,
        }
        self.person_employee[person_id] = result
        print(f"[FaceID] person_id {person_id} dikenali sebagai {match.employee_name} "
              f"(score={match.score:.3f}, margin={match.margin:.3f})")
        return result

    def process_frame(self, frame_path: str):
        """Proses SATU frame baru. Dipanggil tiap kali stream_capture.py
        (atau file-watcher-nya) mendapati 1 file jpg baru selesai ditulis."""
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"[WARN] Gagal baca frame: {frame_path}")
            return None

        if self.csv_file is None:
            self.csv_file = open(self.violation_log_path, "w", newline="")
            self.writer = csv.DictWriter(
                self.csv_file,
                fieldnames=["frame", "frame_file", "track_id", "person_id", "in_zone",
                            "raw_activity", "smoothed_activity", "is_confirmed_violation",
                            "violation_timestamp", "frame_timestamp",
                            "employee_id", "employee_name", "employee_score"]
            )
            self.writer.writeheader()

        # OCR timestamp cuma dijalankan sekali, dari frame PERTAMA yang datang
        if self.video_base_dt is None:
            raw_ts = extract_cctv_timestamp(frame)
            if raw_ts:
                try:
                    self.video_base_dt = datetime.strptime(raw_ts, VIDEO_TIMESTAMP_FORMAT)
                    print(f"[OCR] Base timestamp video (frame 1): {raw_ts}")
                except ValueError:
                    print(f"[OCR] Warning: gagal parse '{raw_ts}', violation timestamp akan '-' untuk video ini")
            else:
                print("[OCR] WARNING: gagal ekstrak timestamp dari frame pertama,"
                      "violation_timestamp akan '-' untuk seluruh video ini")

        self.frame_count += 1
        frame_count = self.frame_count
        frame_filename = os.path.basename(frame_path)

        current_frame_timestamp = compute_frame_timestamp(self.video_base_dt, frame_count, FRAME_PER_SECOND) or "-"

        # ── Cleanup track_id yang sudah lewat grace period ──
        for tid in list(self.track_last_seen.keys()):
            if frame_count - self.track_last_seen[tid] > GRACE_PERIOD_FRAMES:
                del self.track_last_seen[tid]
                pid = self.tid_to_person.pop(tid, None)
                if pid is not None:
                    self.person_active_tid.pop(pid, None)
                for state in [self.activity_history, self.violation_counter, self.last_smoothed,
                              self.last_confirmed, self.violation_timestamp]:
                    state.pop(tid, None)

        # ── Cleanup person_id yang sudah lewat batas re-id ──
        for pid in list(self.person_last_seen_frame.keys()):
            if pid in self.person_active_tid:
                continue
            last_seen = self.person_last_seen_frame.get(pid)
            if last_seen is None:
                continue
            if frame_count - last_seen > REID_MAX_GAP_FRAMES:
                if pid in self.video_trackers:
                    self.video_trackers[pid].finalize()
                    del self.video_trackers[pid]
                for state in [self.person_last_embedding, self.person_last_position,
                              self.person_last_seen_frame, self.person_clip_count,
                              self.person_employee, self.person_face_attempts,
                              self.person_last_face_try]:
                    state.pop(pid, None)

        if self.frame_size is None:
            h, w = frame.shape[:2]
            self.frame_size = (w, h)

            # self.full_video_writer = cv2.VideoWriter(
            #     self.annotated_video_path, VIDEO_FOURCC, FRAME_PER_SECOND, self.frame_size
            # )
            # print(f"[Inference] Full annotated video -> {self.annotated_video_path}")

        # ── Jalankan YOLO ──
        yolo_results = yolo_model.track(frame, conf=0.35, persist=True, tracker="bytetrack.yaml", verbose=False)
        boxes = yolo_results[0].boxes

        if boxes.id is None:
            # if self.full_video_writer is not None:
            #     self.full_video_writer.write(frame)

            if frame_count % 30 == 0:
                print(f"[Inference] Progress: {frame_count} frame diproses")

            self.csv_file.flush()
            return {"frame_count": frame_count, "violation_detected": False}

        current_boxes = [tuple(map(int, b)) for b in boxes.xyxy.cpu().numpy().astype(int)]
        current_tids = [int(t) for t in boxes.id.cpu().numpy().astype(int)]

        print(f"[DEBUG] frame {frame_count}: current_boxes={len(current_boxes)}, tids={current_tids}")

        frame_has_violation = False
        confirmed_tids_this_frame = set()
        in_zone_tids_this_frame = set()
        newly_matched_this_frame = []   # [(person_id, employee_id, score), ...] -- untuk resolusi konflik §6.7

        for ci, bbox in enumerate(current_boxes):
            x1, y1, x2, y2 = bbox

            tid = current_tids[ci]
            self.track_last_seen[tid] = frame_count

            foot_x = (x1 + x2) // 2
            foot_y = y2
            in_zone = zone.contains(Point(foot_x, foot_y))

            if in_zone:
                in_zone_tids_this_frame.add(tid)

            # ── OUT OF ZONE: skip CLIP ──
            if not in_zone:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                cv2.putText(frame, f"ID:{tid} OUT",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

                pid_out = self.tid_to_person.get(tid, "-")
                cached_employee = self.person_employee.get(pid_out) if isinstance(pid_out, int) else None
                cached_employee = cached_employee if isinstance(cached_employee, dict) else None

                self.writer.writerow({
                    "frame"                 : frame_count,
                    "frame_file"            : frame_filename,
                    "track_id"              : tid,
                    "person_id": pid_out,
                    "in_zone"               : False,
                    "raw_activity"          : "-",
                    "smoothed_activity"     : "-",
                    "is_confirmed_violation": False,
                    "violation_timestamp"   : "-",
                    "frame_timestamp": current_frame_timestamp,
                    "employee_id": cached_employee["employee_id"] if cached_employee else "-",
                    "employee_name": cached_employee["employee_name"] if cached_employee else "-",
                    "employee_score": f"{cached_employee['score']:.3f}" if cached_employee else "-",
                })
                continue

            # ── IN ZONE: jalankan CLIP ──
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_image   = Image.fromarray(crop_rgb)
            image_input = clip_preprocess(pil_image).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features  = clip_model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                similarities    = (image_features @ activity_features.T).squeeze(0)
                similarities    = similarities.cpu().numpy()

            best_idx  = int(np.argmax(similarities))
            raw_label = ACTIVITIES[best_idx]

            embedding_np = image_features.squeeze(0).cpu().numpy()

            if tid not in self.tid_to_person:
                person_id = self._resolve_person_id(tid, embedding_np, (foot_x, foot_y), frame_count)
                self.tid_to_person[tid] = person_id
                self.person_active_tid[person_id] = tid
            else:
                person_id = self.tid_to_person[tid]

            if tid not in self.activity_history:
                self.activity_history[tid]    = deque(maxlen=SMOOTHING_WINDOW)
                self.violation_counter[tid]   = 0
                self.last_smoothed[tid]       = "-"
                self.last_confirmed[tid]      = False
                self.violation_timestamp[tid] = None

            self.person_last_embedding[person_id] = embedding_np
            self.person_last_position[person_id] = (foot_x, foot_y)
            self.person_last_seen_frame[person_id] = frame_count

            # ── FACE IDENTIFICATION (Fase 3) ──
            was_cached = person_id in self.person_employee
            employee = self._resolve_employee_id(person_id, crop, frame_count)
            if employee is not None and not was_cached:
                newly_matched_this_frame.append((person_id, employee["employee_id"], employee["score"]))

            # ── MAJORITY VOTING ──
            self.activity_history[tid].append(raw_label)
            smoothed_label = Counter(self.activity_history[tid]).most_common(1)[0][0]
            self.last_smoothed[tid] = smoothed_label

            # ── VIOLATION CONFIRMATION ──
            if smoothed_label in VIOLATIONS:
                self.violation_counter[tid] += 1
            else:
                self.violation_counter[tid] = 0

            is_confirmed = self.violation_counter[tid] >= VIOLATION_CONFIRM_FRAMES

            if is_confirmed and not self.last_confirmed[tid]:
                self.violation_timestamp[tid] = compute_frame_timestamp(
                    self.video_base_dt, frame_count, FRAME_PER_SECOND
                )
                print(f"[Timestamp] Track {tid} violation terkonfirmasi pada: {self.violation_timestamp[tid]}")

            self.last_confirmed[tid] = is_confirmed

            if is_confirmed:
                confirmed_tids_this_frame.add(tid)

            # ── RENDER ──
            clean_label = smoothed_label.replace("a person", "").strip()
            box_color   = (0, 0, 255) if is_confirmed else (0, 255, 0)
            status_text = "VIOLATION" if is_confirmed else "COMPLIANT"

            id_text = employee["employee_name"] if employee is not None else f"ID:{tid}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.circle(frame, (foot_x, foot_y), 5, box_color, -1)
            cv2.putText(frame, id_text,
                        (x1, y1 - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
            draw_label(frame, clean_label, status_text, x1, y1, x2, y2, box_color)

            # ── LOG ──
            log_entry = {
                "frame"                 : frame_count,
                "frame_file"            : frame_filename,
                "track_id"              : tid,
                "person_id": person_id,
                "in_zone"               : True,
                "raw_activity"          : raw_label,
                "smoothed_activity"     : smoothed_label,
                "is_confirmed_violation": is_confirmed,
                "violation_timestamp"   : self.violation_timestamp[tid] or "-",
                "frame_timestamp": current_frame_timestamp,
                "employee_id": employee["employee_id"] if employee else "-",
                "employee_name": employee["employee_name"] if employee else "-",
                "employee_score": f"{employee['score']:.3f}" if employee else "-",
            }
            self.results_log.append(log_entry)
            self.writer.writerow(log_entry)

            if is_confirmed:
                frame_has_violation = True

        # ── RESOLUSI KONFLIK IDENTITAS (Fase 3, plan §6.7) ──
        # Kalau >1 person_id berbeda baru match ke employee_id yang SAMA di frame
        # ini, pertahankan yang skornya tertinggi, lepas kunci yang lain supaya
        # bisa dicoba ulang di frame berikutnya (bukan dikunci UNKNOWN permanen).
        by_employee: dict[str, list[tuple[int, float]]] = {}
        for pid, employee_id, score in newly_matched_this_frame:
            by_employee.setdefault(employee_id, []).append((pid, score))

        for employee_id, matches in by_employee.items():
            if len(matches) <= 1:
                continue
            matches.sort(key=lambda m: m[1], reverse=True)
            for pid, score in matches[1:]:
                print(f"[FaceID] Konflik: person_id {pid} juga match ke {employee_id} "
                      f"(score={score:.3f}), skor person_id lain lebih tinggi -- kunci dilepas")
                self.person_employee.pop(pid, None)

        self.max_concurrent_persons = max(self.max_concurrent_persons, len(in_zone_tids_this_frame))

        person_ids_this_frame = {self.tid_to_person[t] for t in in_zone_tids_this_frame if t in self.tid_to_person}
        for pid in person_ids_this_frame:
            if pid not in self.video_trackers and self.frame_size is not None:
                self.video_trackers[pid] = ViolationVideoTracker(pid, self.frame_size, self.violation_video_dir)

        confirmed_pids_this_frame = {
            self.tid_to_person[t] for t in confirmed_tids_this_frame if t in self.tid_to_person
        }

        for pid, vt in self.video_trackers.items():
            is_viol = pid in confirmed_pids_this_frame
            vt.push(frame, is_viol)

        if frame_has_violation:
            violation_frame_path = os.path.join(self.violation_frames_dir, f"frame{frame_count:06d}.jpg")
            cv2.imwrite(violation_frame_path, frame)

        # if self.full_video_writer is not None:
        #     self.full_video_writer.write(frame)

        if frame_count % 30 == 0:
            print(f"[Inference] Progress: {frame_count} frame diproses")

        # flush supaya baris CSV langsung terlihat di disk begitu ditulis,
        # tanpa menunggu file ditutup di finalize()
        self.csv_file.flush()

        return {"frame_count": frame_count, "violation_detected": frame_has_violation}

    def finalize(self):
        """Dipanggil 1x setelah frame terakhir selesai diproses. Menutup
        video/CSV writer, konversi video ke H.264, dan hitung statistik akhir
        — persis bagian penutup run_inference() yang lama."""
        for tid, vt in self.video_trackers.items():
            vt.finalize()

        # if self.full_video_writer is not None:
        #     self.full_video_writer.release()
        #     print(f"[Inference] Full annotated video selesai (raw) -> {self.annotated_video_path}")

        #     tmp_path = self.annotated_video_path.replace(".mp4", "_tmp.mp4")
        #     os.rename(self.annotated_video_path, tmp_path)
        #     subprocess.run([
        #         "ffmpeg", "-y",
        #         "-i", tmp_path,
        #         "-vcodec", "libx264",
        #         "-crf", "23",
        #         "-preset", "fast",
        #         self.annotated_video_path
        #     ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        #     os.remove(tmp_path)
        #     print(f"[Inference] Konversi H.264 selesai -> {self.annotated_video_path}")

        if self.csv_file is not None:
            self.csv_file.close()

        if self.frame_count == 0:
            print(f"[Inference] Tidak ada frame yang diproses untuk sesi ini: {self.output_dir}")
            return {
                "violation_log_path": None,
                "Annotated_video_path": None,
                "Total_violations": 0,
                "Total_detections": 0,
                "Max_concurrent_persons": 0,
                "Output_dir": self.output_dir,
                "Identified_employees": [],
                "Unidentified_persons": 0,
                "Identification_rate": 0.0,
            }

        in_zone_logs = [r for r in self.results_log if r["in_zone"]]
        violations   = [r for r in self.results_log if r["is_confirmed_violation"]]

        matched_person_ids = [pid for pid, v in self.person_employee.items() if isinstance(v, dict)]
        unknown_person_ids = [pid for pid, v in self.person_employee.items() if v == "UNKNOWN"]
        identified_employees = sorted({self.person_employee[pid]["employee_name"] for pid in matched_person_ids})
        total_resolved = len(matched_person_ids) + len(unknown_person_ids)
        identification_rate = (len(matched_person_ids) / total_resolved * 100) if total_resolved else 0.0

        print("\n" + "=" * 50)
        print("DONE")
        print("=" * 50)
        print(f"Violation frames : {self.violation_frames_dir}/")
        print(f"Violation log    : {self.violation_log_path}")
        print(f"Total deteksi    : {len(self.results_log)}")
        print(f"Total violations : {len(violations)}")

        unique_tids = sorted(set(r['track_id'] for r in in_zone_logs))
        unique_persons = sorted(set(r['person_id'] for r in in_zone_logs))
        print(f"\nTotal track_id unik (mentah, sebelum re-id) : {len(unique_tids)} -> {unique_tids}")
        print(f"Total person_id unik (setelah re-id)         : {len(unique_persons)} -> {unique_persons}")
        print(f"Max orang in-zone SEKALIGUS di 1 frame (independen re-id): {self.max_concurrent_persons}")

        print(f"\nIdentifikasi karyawan (Fase 3):")
        print(f"  Karyawan teridentifikasi (unik) : {len(identified_employees)} -> {identified_employees}")
        print(f"  Person_id tidak teridentifikasi : {len(unknown_person_ids)}")
        print(f"  Identification rate             : {identification_rate:.1f}% "
              f"({len(matched_person_ids)}/{total_resolved} person_id)")

        print("\nBreakdown aktivitas IN_ZONE (smoothed):")
        activity_counts = Counter(r["smoothed_activity"] for r in in_zone_logs)
        for activity, count in activity_counts.most_common():
            clean  = activity.replace("a person ", "").strip()
            marker = "  ← VIOLATION" if activity in VIOLATIONS else ""
            print(f"  {clean}: {count}x{marker}")

        print("\nBreakdown per Track ID:")
        track_ids = sorted(set(r["track_id"] for r in in_zone_logs))
        for tid in track_ids:
            tid_logs  = [r for r in in_zone_logs if r["track_id"] == tid]
            tid_viols = [r for r in tid_logs if r["is_confirmed_violation"]]
            most_common = Counter(r["smoothed_activity"] for r in tid_logs).most_common(1)
            dominant    = most_common[0][0].replace("a person ", "") if most_common else "-"
            print(f"  ID {tid}: {len(tid_logs)} frame IN_ZONE | "
                  f"dominant: {dominant} | violations: {len(tid_viols)}")

        return {
            "violation_log_path": self.violation_log_path,
            "Annotated_video_path": self.annotated_video_path,
            "Total_violations": len(violations),
            "Total_detections": len(self.results_log),
            "Max_concurrent_persons": self.max_concurrent_persons,
            "Output_dir": self.output_dir,
            "Identified_employees": identified_employees,
            "Unidentified_persons": len(unknown_person_ids),
            "Identification_rate": round(identification_rate, 1),
        }


# ─────────────────────────────────────────────
#  ADAPTER BATCH — supaya bisa dites/dipakai persis seperti run_inference()
#  lama (baca semua frame dari 1 folder, sekali panggil), tapi di baliknya
#  tetap lewat InferenceSession yang sama dengan yang dipakai jalur streaming.
#  Ini BUKAN bagian wajib dari desain streaming, cuma pembanding/kompatibilitas.
# ─────────────────────────────────────────────

def run_inference(frame_folder: str, output_dir: str):
    session = InferenceSession(output_dir)

    frame_files = sorted(glob.glob(os.path.join(frame_folder, "*.jpg")))
    if not frame_files:
        print(f"[Inference] Tidak ada frame di folder: {frame_folder}")
        return session.finalize()

    print(f"\n[Inference] Frame folder : {frame_folder}")
    print(f"[Inference] Total frame  : {len(frame_files)}")
    print(f"[Inference] Pre-buffer: {PRE_VIOLATIONS_FRAME} frame ({PRE_VIOLATION_DURATION}) detik")
    print(f"[Inference] Post-buffer: {POST_VIOLATION_FRAME} frame ({POST_VIOLATION_DURATION}) detik")
    print("[Inference] Processing...\n")

    for frame_path in frame_files:
        session.process_frame(frame_path)

    return session.finalize()


if __name__ == "__main__":
    print(run_inference(frame_folder=INPUT_PATH, output_dir=OUTPUT_PATH))
