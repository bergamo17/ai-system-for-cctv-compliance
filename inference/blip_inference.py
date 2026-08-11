import os
import csv
import glob
import cv2
#import torch
#import open_clip
import subprocess
import time
import base64
from io import BytesIO
from openai import OpenAI
import numpy as np
from PIL import Image
from datetime import datetime
from collections import deque, Counter
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
from config import FRAME_INTERVAL, FRAME_PER_SECOND, PRE_VIOLATION_DURATION, POST_VIOLATION_DURATION, HF_TOKEN, MODEL


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

INPUT_PATH           = "frames"
OUTPUT_PATH          = "output"
VIOLATION_FRAMES_DIR = "output/frames/violation"
VIOLATION_VIDEO_DIR = "output/videos/violation"
COMPLIANT_FRAMES_DIR = "output/frames/compliant"


PRE_VIOLATIONS_FRAME = max(1, int(PRE_VIOLATION_DURATION * FRAME_PER_SECOND))
POST_VIOLATION_FRAME = max(1, int(POST_VIOLATION_DURATION * FRAME_PER_SECOND))

SMOOTHING_WINDOW        = 10   # majority voting dari N inferensi terakhir per track
VIOLATION_CONFIRM_FRAMES = 1  # violation dikonfirmasi setelah N inferensi violation berturut-turut
IOU_THRESHOLD           = 0.4  # minimum IoU untuk mencocokkan bbox ke track yang sama
GRACE_PERIOD_FRAMES     = 30   # frame sebelum track dihapus jika tidak terdeteksi

VIDEO_FOURCC = cv2.VideoWriter_fourcc(*'mp4v')

ZONE_POLYGON = [
    (29, 2),
    (398, 136),
    (244, 541),
    (2, 182)
]

ACTIVITIES = [
    'a person preparing or serving a drink',
    'a person standing behind a counter',
    'a person talking to a customer at a counter',
    'a person cleaning or organizing the counter',
    'a person walking behind the counter',
    'a person sitting idle doing nothing',
    'a person using a phone',
    'a person eating food',
    'a person lying down sleeping',
]

# VIOLATIONS harus selalu subset dari ACTIVITIES
VIOLATIONS = [
    'a person using a phone',
    'a person eating food',
    'a person lying down sleeping',
    'a person sitting idle doing nothing',
]

vision_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)


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
#  HELPER: IoU
# ─────────────────────────────────────────────

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter / float(areaA + areaB - inter)


# ─────────────────────────────────────────────
#  HELPER: MATCH BOXES → TRACK IDs
# ─────────────────────────────────────────────

def match_boxes(prev_tracks, current_boxes):
    matched   = [None] * len(current_boxes)
    used_tids = set()

    for ci, cbox in enumerate(current_boxes):
        best_iou = 0.0
        best_tid = None

        for tid, tbox in prev_tracks.items():
            if tid in used_tids:
                continue
            iou = compute_iou(cbox, tbox)
            if iou > best_iou:
                best_iou = iou
                best_tid = tid

        if best_iou >= IOU_THRESHOLD and best_tid is not None:
            matched[ci] = best_tid
            used_tids.add(best_tid)

    return matched


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
            self.current_clipname, VIDEO_FOURCC, FRAME_INTERVAL, (w, h)
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
yolo_model = YOLO("yolov8s-worldv2.pt")
yolo_model.set_classes(['person'])


zone = Polygon(ZONE_POLYGON)


# ─────────────────────────────────────────────
#  ENTRY POINT — dipanggil dari watcher
# ─────────────────────────────────────────────

def encode_crop_to_base64(crop_bgr) -> str:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(crop_rgb)
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=80)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def classify_activity(crop_bgr, activities: list[str], max_retries: int = 2) -> str:
    b64 = encode_crop_to_base64(crop_bgr)
    option_next = "\n".join(f"{i+1}. {act}" for i, act in enumerate(activities))

    prompt_text = (
        
    )


def run_inference(frame_folder: str):
    print(f"[DEBUG] PRE_VIOLATIONS_FRAME={PRE_VIOLATIONS_FRAME}, POSt_VIOLATION_FRAME={POST_VIOLATION_FRAME}")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(VIOLATION_VIDEO_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    violation_log_path = os.path.join(OUTPUT_PATH, f"violation_log_{timestamp}.csv")

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

    # ── Tracking state — reset tiap video baru ──
    next_track_id    = [0]
    active_tracks = {}
    track_last_seen  = {}
    activity_history = {}
    violation_counter= {}
    last_smoothed    = {}
    last_confirmed   = {}

    video_trackers: dict[int, ViolationVideoTracker] = {}

    frame_count = 0
    results_log = []
    frame_size = None

    with open(violation_log_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["frame", "frame_file", "track_id", "in_zone",
                        "raw_activity", "smoothed_activity", "is_confirmed_violation"]
        )
        writer.writeheader()

        for frame_path in frame_files:
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"[WARN] Gagal baca frame: {frame_path}")
                continue

            frame_count += 1
            frame_filename = os.path.basename(frame_path)

            if frame_size is None:
                h, w = frame.shape[:2]
                frame_size = (w, h)

            # ── Jalankan YOLO ──
            yolo_results  = yolo_model(frame, conf=0.35, verbose=False)
            current_boxes = [tuple(map(int, b.xyxy[0])) for b in yolo_results[0].boxes]

            matched_tids = match_boxes(active_tracks, current_boxes)
            print(f"[DEBUG] frame {frame_count}: current_boxes={len(current_boxes)}, matched={matched_tids}")

            #new_track_boxes     = {}
            frame_has_violation = False

            confirmed_tids_this_frame = set()

            for ci, bbox in enumerate(current_boxes):
                x1, y1, x2, y2 = bbox

                # Assign atau buat track ID
                tid = matched_tids[ci]
                if tid is None:
                    tid              = next_track_id[0]
                    next_track_id[0] += 1

                active_tracks[tid] = bbox
                track_last_seen[tid] = frame_count

                # Inisialisasi state track baru
                if tid not in activity_history:
                    activity_history[tid]  = deque(maxlen=SMOOTHING_WINDOW)
                    violation_counter[tid] = 0
                    last_smoothed[tid]     = "-"
                    last_confirmed[tid]    = False

                foot_x  = (x1 + x2) // 2
                foot_y  = y2
                in_zone = zone.contains(Point(foot_x, foot_y))

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
                        "in_zone"               : False,
                        "raw_activity"          : "-",
                        "smoothed_activity"     : "-",
                        "is_confirmed_violation": False,
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

                # ── MAJORITY VOTING ──
                activity_history[tid].append(raw_label)
                smoothed_label     = Counter(activity_history[tid]).most_common(1)[0][0]
                last_smoothed[tid] = smoothed_label

                # ── VIOLATION CONFIRMATION ──
                if smoothed_label in VIOLATIONS:
                    violation_counter[tid] += 1
                else:
                    violation_counter[tid] = 0

                is_confirmed        = violation_counter[tid] >= VIOLATION_CONFIRM_FRAMES
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
                    "in_zone"               : True,
                    "raw_activity"          : raw_label,
                    "smoothed_activity"     : smoothed_label,
                    "is_confirmed_violation": is_confirmed,
                }
                results_log.append(log_entry)
                writer.writerow(log_entry)

                if is_confirmed:
                    frame_has_violation = True

            for tid in active_tracks:
                if tid not in video_trackers and frame_size is not None:
                    video_trackers[tid] = ViolationVideoTracker(tid, frame_size, VIOLATION_VIDEO_DIR)

            for tid, vt in video_trackers.items():
                is_viol = tid in confirmed_tids_this_frame
                vt.push(frame, is_viol)

            # ── GRACE PERIOD: hapus track lama ──
            for tid in list(active_tracks.keys()):
                if frame_count - track_last_seen.get(tid, 0) > GRACE_PERIOD_FRAMES:
                    if tid in video_trackers:
                        video_trackers[tid].finalize()
                        del video_trackers[tid]
                        
                    del active_tracks[tid]
                    for state in [activity_history, violation_counter, last_smoothed, last_confirmed]:
                        if tid in state:
                            del state[tid]

            
            if frame_count % 30 == 0:
                pct = (frame_count / total_frames) * 100
                print(f"[Inference] Progress: {frame_count}/{total_frames} frame ({pct:.1f}%)")

    for tid, vt in video_trackers.items():
        vt.finalize()
    
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