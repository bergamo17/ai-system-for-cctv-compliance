import cv2
import pickle
import os
from config import TIMESTAMP_CROP
FRAME_PATH   = "frames/frame_000001.jpg"
GROUND_TRUTH = "05-08-2026 14:23:07"   # HARUS sama persis dengan yang tampil di overlay
TEMPLATE_PATH = "digit_templates.pkl"
TIMESTAMP_CROP = [(6, 47), (307, 47), (306, 112), (4, 115)]
def segment_characters(thresh_img, min_w=2, min_h=8):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = [b for b in boxes if b[2] >= min_w and b[3] >= min_h]
    boxes.sort(key=lambda b: b[0])  # urut kiri -> kanan
    return boxes
def main():
    frame = cv2.imread(FRAME_PATH)
    if frame is None:
        print(f"[ERROR] Gagal baca {FRAME_PATH}")
        return
    xs = [p[0] for p in TIMESTAMP_CROP]
    ys = [p[1] for p in TIMESTAMP_CROP]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    roi = frame[y1:y2, x1:x2]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.resize(roi_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, roi_thresh = cv2.threshold(roi_gray, 180, 255, cv2.THRESH_BINARY)
    boxes = segment_characters(roi_thresh)
    gt_chars = [c for c in GROUND_TRUTH if c != " "]  # buang spasi pemisah tanggal/jam
    if len(boxes) != len(gt_chars):
        print(f"[WARN] jumlah box tersegmentasi ({len(boxes)}) != panjang ground truth ({len(gt_chars)})")
        print("       Cek debug_char_*.png setelah ini, kemungkinan noise atau threshold kurang pas.")
    # load template lama kalau ada, biar bisa akumulasi antar-run
    templates = {}
    if os.path.exists(TEMPLATE_PATH):
        with open(TEMPLATE_PATH, "rb") as f:
            templates = pickle.load(f)
    for i, (bx, by, bw, bh) in enumerate(boxes):
        if i >= len(gt_chars):
            break
        crop = roi_thresh[by:by+bh, bx:bx+bw]
        ch = gt_chars[i]
        templates[ch] = crop
        safe_name = {"-": "dash", ":": "colon"}.get(ch, ch)
        cv2.imwrite(f"debug_char_{i:02d}_{safe_name}.png", crop)
    with open(TEMPLATE_PATH, "wb") as f:
        pickle.dump(templates, f)
    needed = set("0123456789-:")
    missing = needed - set(templates.keys())
    print(f"Saved {len(templates)} templates -> {TEMPLATE_PATH}")
    print(f"Karakter ter-cover: {sorted(templates.keys())}")
    if missing:
        print(f"[INFO] Masih kurang: {sorted(missing)} -> cari frame lain yang punya digit ini, run lagi")
if __name__ == "__main__":
    main()