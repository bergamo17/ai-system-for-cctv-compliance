import os
import re
import csv
import glob
import cv2
import torch
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
#  CONFIGURATION
# ─────────────────────────────────────────────

INPUT_PATH           = "frames"
OUTPUT_PATH          = "output"
VIOLATION_FRAMES_DIR = "output/frames/violation"
VIOLATION_VIDEO_DIR = "output/videos/violation"
COMPLIANT_FRAMES_DIR = "output/frames/compliant"

ANNOTATED_VIDEO_DIR = "output/videos/annotated"


PRE_VIOLATIONS_FRAME = max(1, int(PRE_VIOLATION_DURATION * FRAME_PER_SECOND))
POST_VIOLATION_FRAME = max(1, int(POST_VIOLATION_DURATION * FRAME_PER_SECOND))

SMOOTHING_WINDOW        = 3   # majority voting dari N inferensi terakhir per track
VIOLATION_CONFIRM_FRAMES = 5  # violation dikonfirmasi setelah N inferensi violation berturut-turut
IOU_THRESHOLD           = 0.15  # minimum IoU untuk mencocokkan bbox ke track yang sama
GRACE_PERIOD_FRAMES     = 30   # frame sebelum track dihapus jika tidak terdeteksi

REID_ENABLED = True
REID_MAX_GAP_FRAMES = 90

REID_MIN_COSINE_SCORE = 0.9
REID_MAX_DISTANCE_PX = 150

VIDEO_FOURCC = cv2.VideoWriter_fourcc(*'mp4v')

VIDEO_TIMESTAMP_FORMAT = "%d-%m-%Y %H:%M:%S"

DIGIT_TEMPLATES = {}
if os.path.exists("digit_templates.pkl"):
    with open("digit_templates.pkl", "rb") as f:
        DIGIT_TEMPLATES = pickle.load(f)
else:
    print("[WARN] digit_templates.pkl tidak ditemukan, template matching akan selalu fallback ke Tesseract")

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
            return None  # satu karakter aja gagal match, batalkan semua
        result_chars.append(ch)

    raw = "".join(result_chars)
    # susun ulang jadi "DD-MM-YYYY HH:MM:SS" (masukin spasi kembali di posisi ke-10)
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

    # upscale biar OCR lebih akurat untuk teks kecil
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

ZONE_POLYGON = [
    (462, 1079),
    (0, 243),
    (168, 2),
    (792, 265)
]


# ─────────────────────────────────────────────
#  HELPER: DRAW LABEL
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
#  HELPER: Re-ID
# ─────────────────────────────────────────────

def cosine_sim(emb_a, emb_b):
    return float(np.dot(emb_a, emb_b))

def euclidean_dist(pos_a, pos_b):
    return float(np.hypot(pos_a[0]-pos_b[0], pos_a[1]-pos_b[1]))


# ─────────────────────────────────────────────
#  HELPER: IoU
# ─────────────────────────────────────────────

# def compute_iou(boxA, boxB):
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])

#     inter = max(0, xB - xA) * max(0, yB - yA)
#     if inter == 0:
#         return 0.0

#     areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

#     return inter / float(areaA + areaB - inter)


# ─────────────────────────────────────────────
#  HELPER: MATCH BOXES → TRACK IDs
# ─────────────────────────────────────────────

# def match_boxes(prev_tracks, current_boxes):
#     matched   = [None] * len(current_boxes)
#     used_tids = set()

#     for ci, cbox in enumerate(current_boxes):
#         best_iou = 0.0
#         best_tid = None

#         for tid, tbox in prev_tracks.items():
#             if tid in used_tids:
#                 continue
#             iou = compute_iou(cbox, tbox)
#             if iou > best_iou:
#                 best_iou = iou
#                 best_tid = tid

#         if best_iou >= IOU_THRESHOLD and best_tid is not None:
#             matched[ci] = best_tid
#             used_tids.add(best_tid)

#     return matched


# ─────────────────────────────────────────────
#  LOAD MODELS — dilakukan sekali saat import
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

        # elif self.state == 'POST_BUFFER':
        #     self.writer.write(annotated_frame.copy())
        #     self.post_count += 1

        #     if self.post_count >= POST_VIOLATION_FRAME:
        #         self._close_writer()
        #         self.pre_buffer.clear()
        #         self.state = 'IDLE'

        #     elif is_confirmed_violation:
        #         self.state = 'RECORDING'
        #         self.post_count = 0

    def finalize(self):
        if self.state in ('RECORDING', 'POST_BUFFER') and self.writer:
            self._close_writer()


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

zone = Polygon(ZONE_POLYGON)


# ─────────────────────────────────────────────
#  ENTRY POINT — dipanggil dari watcher
# ─────────────────────────────────────────────

def run_inference(frame_folder: str):
    print(f"[DEBUG] PRE_VIOLATIONS_FRAME={PRE_VIOLATIONS_FRAME}, POSt_VIOLATION_FRAME={POST_VIOLATION_FRAME}")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(VIOLATION_VIDEO_DIR, exist_ok=True)
    os.makedirs(ANNOTATED_VIDEO_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    violation_log_path = os.path.join(OUTPUT_PATH, f"violation_log_{timestamp}.csv")

    annotated_video_path = os.path.join(ANNOTATED_VIDEO_DIR, f"annotated_{timestamp}.mp4")
    full_video_writer = None

    # ── Baca frame dari folder ──
    frame_files = sorted(glob.glob(os.path.join(frame_folder, "*.jpg")))

    if not frame_files:
        print(f"[Inference] Tidak ada frame di folder: {frame_folder}")
        return

    total_frames = len(frame_files)
    print(f"\n[Inference] Frame folder : {frame_folder}")
    print(f"[Inference] Total frame  : {total_frames}")
    print(f"[Inference] Pre-buffer: {PRE_VIOLATIONS_FRAME} frame ({PRE_VIOLATION_DURATION}) detik")
    print(f"[Inference] Post-buffer: {POST_VIOLATION_FRAME} frame ({POST_VIOLATION_DURATION}) detik")
    print("[Inference] Processing...\n")

    video_base_dt = None
    first_frame = cv2.imread(frame_files[0])
    if first_frame is not None:
        raw_ts = extract_cctv_timestamp(first_frame)
        if raw_ts:
            try:
                video_base_dt = datetime.strptime(raw_ts, VIDEO_TIMESTAMP_FORMAT)
                print(f"[OCR] Base timestamp video (frame 1): {raw_ts}")
            except ValueError:
                print(f"[OCR] Warning: gagal parse '{raw_ts}', violation timestamp akan '-' untuk video ini")
        else:
            print("[OCR] WARNING: gagal ekstrak timestamp dari frame pertama,"
                    "violation_timestamp akan '-' untuk seluruh video ini")
    else:
        print(f"[WARN] Gagal baca frame pertama untuk OCR: {frame_files[0]}")

    # ── Tracking state — reset tiap video baru ──
    # Dict berikut semuanya di-index pakai track_id (tid), karena history
    # aktivitas & konfirmasi violation itu per-track (hasil YOLO/bytetrack),
    # bukan per-person (hasil re-id).
    track_last_seen      = {}
    activity_history     = {}
    violation_counter    = {}
    last_smoothed        = {}
    last_confirmed       = {}
    violation_timestamp  = {}   # OCR result, di-index pakai tid

    video_trackers: dict[int, ViolationVideoTracker] = {}
    person_clip_count = {}

    # Dict berikut di-index pakai person_id (pid), hasil dari re-id
    tid_to_person = {}
    person_active_tid = {}
    person_last_embedding = {}
    person_last_position = {}
    person_last_seen_frame = {}
    next_person_id = [1]

    max_concurrent_persons = 0

    frame_count = 0
    results_log = []
    frame_size = None

    def resolve_person_id(tid, embedding, position, frame_count):
        if REID_ENABLED:
            best_pid, best_sim = None, 0.0
            for pid, last_frame in person_last_seen_frame.items():
                if pid in person_active_tid:
                    continue
                if frame_count - last_frame > REID_MAX_GAP_FRAMES:
                    continue
                sim = cosine_sim(embedding, person_last_embedding[pid])
                dist = euclidean_dist(position, person_last_position[pid])
                if sim >= REID_MIN_COSINE_SCORE and dist <= REID_MAX_DISTANCE_PX and sim > best_sim:
                    best_sim, best_pid = sim, pid

            if best_pid is not None:
                print(f"[ReID] track_id {tid} dikenali sebagai person_id {best_pid}"
                      f"(cosine_sim:{best_sim:.3f})")
                return best_pid
            
        pid = next_person_id[0]
        next_person_id[0] += 1
        return pid

    with open(violation_log_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["frame", "frame_file", "track_id", "person_id", "in_zone",
                        "raw_activity", "smoothed_activity", "is_confirmed_violation",
                        "violation_timestamp", "frame_timestamp"]
        )
        writer.writeheader()

        for frame_path in frame_files:
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"[WARN] Gagal baca frame: {frame_path}")
                continue

            frame_count += 1
            frame_filename = os.path.basename(frame_path)

            current_frame_timestamp = compute_frame_timestamp(video_base_dt, frame_count, FRAME_PER_SECOND) or "-"

            # ── Cleanup track_id yang sudah lewat grace period ──
            for tid in list(track_last_seen.keys()):
                if frame_count - track_last_seen[tid] > GRACE_PERIOD_FRAMES:
                    del track_last_seen[tid]
                    pid = tid_to_person.pop(tid, None)
                    if pid is not None:
                        person_active_tid.pop(tid, None)
                    # dict-dict berikut di-index pakai tid, bukan pid,
                    # jadi dibersihkan di sini (bukan di loop cleanup person)
                    for state in [activity_history, violation_counter, last_smoothed,
                                  last_confirmed, violation_timestamp]:
                        state.pop(tid, None)

            # ── Cleanup person_id yang sudah lewat batas re-id ──
            for pid in list(person_last_seen_frame.keys()):
                if pid in person_active_tid:
                    continue  # masih aktif dipegang tid saat ini, jangan disentuh
                last_seen = person_last_seen_frame.get(pid)
                if last_seen is None:
                    continue  # sudah terhapus duluan oleh proses lain, skip saja
                if frame_count - last_seen > REID_MAX_GAP_FRAMES:
                    if pid in video_trackers:
                        video_trackers[pid].finalize()
                        del video_trackers[pid]
                    # dict-dict berikut di-index pakai pid
                    for state in [person_last_embedding, person_last_position,
                                  person_last_seen_frame, person_clip_count]:
                        state.pop(pid, None)

            if frame_size is None:
                h, w = frame.shape[:2]
                frame_size = (w, h)

                full_video_writer = cv2.VideoWriter(
                    annotated_video_path, VIDEO_FOURCC, FRAME_PER_SECOND, frame_size
                )
                print(f"[Inference] Full annotated video -> {annotated_video_path}")

            # ── Jalankan YOLO ──
            yolo_results  = yolo_model.track(frame, conf=0.35, persist=True, tracker="bytetrack.yaml", verbose=False)
            boxes = yolo_results[0].boxes

            if boxes.id is None:
                if full_video_writer is not None:
                    full_video_writer.write(frame)

                if frame_count % 30 == 0:
                    pct = (frame_count / total_frames) * 100
                    print(f"[Inference] Progress: {frame_count}/{total_frames} frame ({pct:.1f}%)")
                continue

            current_boxes = [tuple(map(int, b)) for b in boxes.xyxy.cpu().numpy().astype(int)]
            current_tids = [int(t) for t in boxes.id.cpu().numpy().astype(int)]

            print(f"[DEBUG] frame {frame_count}: current_boxes={len(current_boxes)}, tids={current_tids}")

            frame_has_violation = False

            confirmed_tids_this_frame = set()
            in_zone_tids_this_frame = set()

            for ci, bbox in enumerate(current_boxes):
                x1, y1, x2, y2 = bbox

                # Assign atau buat track ID
                tid = current_tids[ci]
                track_last_seen[tid] = frame_count

                foot_x  = (x1 + x2) // 2
                foot_y  = y2
                in_zone = zone.contains(Point(foot_x, foot_y))

                if in_zone:
                    in_zone_tids_this_frame.add(tid)

                # ── OUT OF ZONE: skip CLIP ──
                if not in_zone:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                    cv2.putText(frame, f"ID:{tid} OUT",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

                    writer.writerow({
                        "frame"                 : frame_count,
                        "frame_file"            : frame_filename,
                        "track_id"              : tid,
                        "person_id": tid_to_person.get(tid, "-"),
                        "in_zone"               : False,
                        "raw_activity"          : "-",
                        "smoothed_activity"     : "-",
                        "is_confirmed_violation": False,
                        "violation_timestamp"   : "-",
                        "frame_timestamp": current_frame_timestamp,
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

                best_idx   = int(np.argmax(similarities))
                raw_label  = ACTIVITIES[best_idx]

                embedding_np = image_features.squeeze(0).cpu().numpy()

                if tid not in tid_to_person:
                    person_id = resolve_person_id(tid, embedding_np, (foot_x, foot_y), frame_count)
                    tid_to_person[tid] = person_id
                    person_active_tid[person_id] = tid
                else:
                    person_id = tid_to_person[tid]

                # Inisialisasi state track baru
                if tid not in activity_history:
                    activity_history[tid]     = deque(maxlen=SMOOTHING_WINDOW)
                    violation_counter[tid]    = 0
                    last_smoothed[tid]        = "-"
                    last_confirmed[tid]       = False
                    violation_timestamp[tid]  = None

                person_last_embedding[person_id] = embedding_np
                person_last_position[person_id] = (foot_x, foot_y)
                person_last_seen_frame[person_id] = frame_count

                # ── MAJORITY VOTING ──
                activity_history[tid].append(raw_label)
                smoothed_label     = Counter(activity_history[tid]).most_common(1)[0][0]
                last_smoothed[tid] = smoothed_label

                # ── VIOLATION CONFIRMATION ──
                if smoothed_label in VIOLATIONS:
                    violation_counter[tid] += 1
                else:
                    violation_counter[tid] = 0

                is_confirmed = violation_counter[tid] >= VIOLATION_CONFIRM_FRAMES

                # OCR HANYA dipanggil sekali, saat transisi COMPLIANT -> VIOLATION
                if is_confirmed and not last_confirmed[tid]:
                    violation_timestamp[tid] = compute_frame_timestamp(
                        video_base_dt, frame_count, FRAME_PER_SECOND
                    )
                    print(f"[Timestamp] Track {tid} violation terkonfirmasi pada: {violation_timestamp[tid]}")

                last_confirmed[tid] = is_confirmed

                if is_confirmed:
                    confirmed_tids_this_frame.add(tid)

                # ── RENDER ──
                clean_label = smoothed_label.replace("a person", "").strip()
                box_color   = (0, 0, 255) if is_confirmed else (0, 255, 0)
                status_text = "VIOLATION" if is_confirmed else "COMPLIANT"

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.circle(frame, (foot_x, foot_y), 5, box_color, -1)
                cv2.putText(frame, f"ID:{tid}",
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
                    "violation_timestamp"   : violation_timestamp[tid] or "-",
                    "frame_timestamp": current_frame_timestamp,
                }
                results_log.append(log_entry)
                writer.writerow(log_entry)

                if is_confirmed:
                    frame_has_violation = True

            max_concurrent_persons = max(max_concurrent_persons, len(in_zone_tids_this_frame))

            person_ids_this_frame = {tid_to_person[t] for t in in_zone_tids_this_frame if t in tid_to_person}
            for pid in person_ids_this_frame:
                if pid not in video_trackers and frame_size is not None:
                    video_trackers[pid] = ViolationVideoTracker(pid, frame_size, VIOLATION_VIDEO_DIR)

            confirmed_pids_this_frame = {
                tid_to_person[t] for t in confirmed_tids_this_frame if t in tid_to_person
            }

            for pid, vt in video_trackers.items():
                is_viol = pid in confirmed_pids_this_frame
                vt.push(frame, is_viol)

            if full_video_writer is not None:
                full_video_writer.write(frame)

            if frame_count % 30 == 0:
                pct = (frame_count / total_frames) * 100
                print(f"[Inference] Progress: {frame_count}/{total_frames} frame ({pct:.1f}%)")

    for tid, vt in video_trackers.items():
        vt.finalize()

    if full_video_writer is not None:
        full_video_writer.release()
        print(f"[Inference] Full annotated video selesai (raw) -> {annotated_video_path}")

        tmp_path = annotated_video_path.replace(".mp4", "_tmp.mp4")
        os.rename(annotated_video_path, tmp_path)
        subprocess.run([
            "ffmpeg", "-y",
            "-i", tmp_path,
            "-vcodec", "libx264",
            "-crf", "23",
            "-preset", "fast",
            annotated_video_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(tmp_path)
        print(f"[Inference] Konversi H.264 selesai -> {annotated_video_path}")
    
    # ── SELESAI ──
    in_zone_logs = [r for r in results_log if r["in_zone"]]
    violations   = [r for r in results_log if r["is_confirmed_violation"]]

    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)
    print(f"Violation frames : {VIOLATION_FRAMES_DIR}/")
    print(f"Violation log    : {violation_log_path}")
    print(f"Total deteksi    : {len(results_log)}")
    print(f"Total violations : {len(violations)}")

    unique_tids = sorted(set(r['track_id'] for r in in_zone_logs))
    unique_persons = sorted(set(r['person_id'] for r in in_zone_logs))
    print(f"\nTotal track_id unik (mentah, sebelum re-id) : {len(unique_tids)} -> {unique_tids}")
    print(f"Total person_id unik (setelah re-id)         : {len(unique_persons)} -> {unique_persons}")
    print(f"Max orang in-zone SEKALIGUS di 1 frame (independen re-id): {max_concurrent_persons}")
    print("^ ini sinyal paling andal untuk 'jumlah karyawan riil' — kirim ke LLM sebagai konteks eksplisit,")
    print("  bahkan kalau person_id di atas belum 100% sempurna.")

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
        
    return violation_log_path

if __name__ == "__main__":
    print(run_inference(frame_folder=INPUT_PATH))