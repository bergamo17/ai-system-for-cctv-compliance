import os
import csv
import glob
import cv2
import torch
import open_clip
import subprocess
import time
import numpy as np
from PIL import Image
from datetime import datetime
from collections import deque, Counter
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
from transformers import BlipProcessor, BlipForConditionalGeneration
from config import FRAME_INTERVAL, FRAME_PER_SECOND, PRE_VIOLATION_DURATION, POST_VIOLATION_DURATION


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
BLIP_CAPTION_INTERVAL_FRAMES = 10  # BLIP hanya jalan 1x tiap N frame yang dicapture; CLIP tetap jalan tiap frame

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

# ─────────────────────────────────────────────
#  HELPER: DRAW LABEL (mendukung N baris teks)
# ─────────────────────────────────────────────

def draw_label(frame, lines, x1, y1, x2, y2, color):
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness  = 2
    padding    = 4

    heights = [cv2.getTextSize(t, font, font_scale, thickness)[0][1] for t in lines]
    total_h = sum(heights) + padding * (len(lines) + 1)

    if y1 - total_h >= 0:
        y = y1 - padding
        for text, h in zip(reversed(lines), reversed(heights)):
            cv2.putText(frame, text, (x1, y), font, font_scale, color, thickness)
            y -= (h + padding)
    else:
        y = y1 + padding
        for text, h in zip(lines, heights):
            y += h
            cv2.putText(frame, text, (x1 + padding, y), font, font_scale, color, thickness)
            y += padding


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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print("Load OpenCLIP model...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k', device=device
)
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

activity_tokens = clip_tokenizer(ACTIVITIES).to(device)
with torch.no_grad():
    activity_features = clip_model.encode_text(activity_tokens)
    activity_features /= activity_features.norm(dim=-1, keepdim=True)

print("Load BLIP captioning model...")
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME).to(device)
blip_model.eval()

zone = Polygon(ZONE_POLYGON)


# ─────────────────────────────────────────────
#  ENTRY POINT — dipanggil dari watcher
# ─────────────────────────────────────────────

def generate_caption(crop_bgr) -> str:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(crop_rgb)
    inputs = blip_processor(pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_new_tokens=30)
    return blip_processor.decode(out[0], skip_special_tokens=True)


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
    last_caption     = {}

    video_trackers: dict[int, ViolationVideoTracker] = {}

    frame_count = 0
    results_log = []
    frame_size = None

    with open(violation_log_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["frame", "frame_file", "track_id", "in_zone",
                        "raw_activity", "smoothed_activity", "is_confirmed_violation", "caption"]
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
                    last_caption[tid]      = "-"

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
                        "caption"               : "-",
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

                # ── BLIP CAPTION: cukup jalan 1x tiap N frame, sisanya reuse cache ──
                if frame_count % BLIP_CAPTION_INTERVAL_FRAMES == 0:
                    last_caption[tid] = generate_caption(crop)
                caption = last_caption[tid]

                # ── RENDER ──
                clean_label = smoothed_label.replace("a person", "").strip()
                box_color   = (0, 0, 255) if is_confirmed else (0, 255, 0)
                status_text = "VIOLATION" if is_confirmed else "COMPLIANT"
                caption_display = caption if len(caption) <= 60 else caption[:57] + "..."

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.circle(frame, (foot_x, foot_y), 5, box_color, -1)
                cv2.putText(frame, f"ID:{tid}",
                            (x1, y1 - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
                draw_label(frame, [clean_label, status_text, caption_display], x1, y1, x2, y2, box_color)

                # ── LOG ──
                log_entry = {
                    "frame"                 : frame_count,
                    "frame_file"            : frame_filename,
                    "track_id"              : tid,
                    "in_zone"               : True,
                    "raw_activity"          : raw_label,
                    "smoothed_activity"     : smoothed_label,
                    "is_confirmed_violation": is_confirmed,
                    "caption"               : caption,
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
                    for state in [activity_history, violation_counter, last_smoothed, last_confirmed, last_caption]:
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