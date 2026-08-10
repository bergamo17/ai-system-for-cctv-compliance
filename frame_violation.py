import os
import csv
import glob
import cv2
import torch
import open_clip
import numpy as np
from PIL import Image
from collections import deque, Counter
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
from config import FRAME_PER_SECOND


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

INPUT_PATH           = "frames"
OUTPUT_PATH          = "output"
VIOLATION_FRAMES_DIR = "output/frames/violation"
COMPLIANT_FRAMES_DIR = "output/frames/compliant"
VIOLATION_LOG_PATH   = "output/violation_log.csv"

SMOOTHING_WINDOW        = 10   # majority voting dari N inferensi terakhir per track
VIOLATION_CONFIRM_FRAMES = 1  # violation dikonfirmasi setelah N inferensi violation berturut-turut
IOU_THRESHOLD           = 0.4  # minimum IoU untuk mencocokkan bbox ke track yang sama
GRACE_PERIOD_FRAMES     = 30   # frame sebelum track dihapus jika tidak terdeteksi

# Supaya tidak flood ribuan frame identik selama 1 violation event berlangsung lama,
# frame disimpan tiap SAVE_EVERY_N_FRAMES frame selama violation masih confirmed
# (frame pertama saat violation baru terkonfirmasi SELALU disimpan).
SAVE_EVERY_N_FRAMES = max(1, int(FRAME_PER_SECOND))  # default: 1 frame per detik selama violation

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
#  HELPER: SIMPAN FRAME VIOLATION
# ─────────────────────────────────────────────

class ViolationFrameSaver:
    """
    Menggantikan ViolationVideoTracker.
    Menyimpan frame (bukan video) tiap kali violation seorang track terkonfirmasi,
    dengan throttling supaya tidak menyimpan frame yang nyaris identik tiap iterasi.
    """
    def __init__(self, track_id, output_dir):
        self.track_id = track_id
        self.output_dir = output_dir
        self.is_active = False       # sedang dalam episode violation atau tidak
        self.frames_since_last_save = 0
        self.event_index = 0         # penomoran episode violation ke-berapa untuk track ini

    def push(self, annotated_frame, frame_count, is_confirmed_violation: bool, activity_label: str):
        if is_confirmed_violation:
            if not self.is_active:
                # episode violation baru dimulai -> selalu simpan frame pertama
                self.is_active = True
                self.event_index += 1
                self.frames_since_last_save = 0
                self._save(annotated_frame, frame_count, activity_label)
            else:
                self.frames_since_last_save += 1
                if self.frames_since_last_save >= SAVE_EVERY_N_FRAMES:
                    self.frames_since_last_save = 0
                    self._save(annotated_frame, frame_count, activity_label)
        else:
            # episode violation berakhir
            self.is_active = False
            self.frames_since_last_save = 0

    def _save(self, annotated_frame, frame_count, activity_label):
        clean_label = activity_label.replace("a person ", "").strip().replace(" ", "-")
        filename = (
            f"track{self.track_id:03d}_event{self.event_index:02d}"
            f"_frame{frame_count:06d}_{clean_label}.jpg"
        )
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, annotated_frame)
        print(f"[FrameSaver] Track {self.track_id}: violation frame disimpan -> {filename}")


print("Load YOLO model...")
yolo_model = YOLO("yolov8s-worldv2.pt")
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

def run_frame_inference(frame_folder: str):
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(VIOLATION_FRAMES_DIR, exist_ok=True)
    os.makedirs(COMPLIANT_FRAMES_DIR, exist_ok=True)

    # ── Baca frame dari folder ──
    frame_files = sorted(glob.glob(os.path.join(frame_folder, "*.jpg")))

    if not frame_files:
        print(f"[Inference] Tidak ada frame di folder: {frame_folder}")
        return

    total_frames = len(frame_files)
    print(f"\n[Inference] Frame folder : {frame_folder}")
    print(f"[Inference] Total frame  : {total_frames}")
    print(f"[Inference] Simpan violation frame tiap {SAVE_EVERY_N_FRAMES} frame selama violation berlangsung")
    print("[Inference] Processing...\n")

    # ── Tracking state — reset tiap video baru ──
    next_track_id    = [0]
    prev_track_boxes = {}
    track_last_seen  = {}
    activity_history = {}
    violation_counter= {}
    last_smoothed    = {}
    last_confirmed   = {}

    frame_savers: dict[int, ViolationFrameSaver] = {}

    frame_count = 0
    results_log = []

    with open(VIOLATION_LOG_PATH, "w", newline="") as csv_file:
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

            # ── Jalankan YOLO ──
            yolo_results  = yolo_model(frame, conf=0.35, verbose=False)
            current_boxes = [tuple(map(int, b.xyxy[0])) for b in yolo_results[0].boxes]

            matched_tids = match_boxes(prev_track_boxes, current_boxes)
            print(f"[DEBUG] frame {frame_count}: current_boxes={len(current_boxes)}, matched={matched_tids}")

            new_track_boxes = {}
            tid_status_this_frame = {}  # tid -> (is_confirmed, smoothed_label)

            for ci, bbox in enumerate(current_boxes):
                x1, y1, x2, y2 = bbox

                # Assign atau buat track ID
                tid = matched_tids[ci]
                if tid is None:
                    tid              = next_track_id[0]
                    next_track_id[0] += 1

                new_track_boxes[tid] = bbox
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
                    tid_status_this_frame[tid] = (False, "-")
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

                tid_status_this_frame[tid] = (is_confirmed, smoothed_label)

            # ── Compliance Decision → simpan frame jika violation ──
            for tid in new_track_boxes:
                if tid not in frame_savers:
                    frame_savers[tid] = ViolationFrameSaver(tid, VIOLATION_FRAMES_DIR)

            for tid, saver in frame_savers.items():
                is_confirmed, smoothed_label = tid_status_this_frame.get(tid, (False, "-"))
                saver.push(frame, frame_count, is_confirmed, smoothed_label)

            # ── GRACE PERIOD: hapus track lama ──
            for tid in list(prev_track_boxes.keys()):
                if frame_count - track_last_seen.get(tid, 0) > GRACE_PERIOD_FRAMES:
                    if tid in frame_savers:
                        del frame_savers[tid]

                    del prev_track_boxes[tid]
                    for state in [activity_history, violation_counter, last_smoothed, last_confirmed]:
                        if tid in state:
                            del state[tid]

            prev_track_boxes = new_track_boxes

            if frame_count % 30 == 0:
                pct = (frame_count / total_frames) * 100
                print(f"[Inference] Progress: {frame_count}/{total_frames} frame ({pct:.1f}%)")

    # ── SELESAI ──
    in_zone_logs = [r for r in results_log if r["in_zone"]]
    violations   = [r for r in results_log if r["is_confirmed_violation"]]

    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)
    print(f"Violation frames : {VIOLATION_FRAMES_DIR}/")
    print(f"Violation log    : {VIOLATION_LOG_PATH}")
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