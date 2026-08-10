import cv2
import pytesseract
from config import TIMESTAMP_CROP
FRAME_PATH = "/kaggle/working/ai-system-for-cctv-compliance/frames/Footage-sample_frame_000001.jpg"
frame = cv2.imread(FRAME_PATH)
if frame is None:
    print(f"[ERROR] Gagal baca frame: {FRAME_PATH}")
else:
    print(f"[INFO] Frame size: {frame.shape[1]}x{frame.shape[0]} (WxH)")
    xs = [p[0] for p in TIMESTAMP_CROP]
    ys = [p[1] for p in TIMESTAMP_CROP]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    print(f"[INFO] Crop region: x=({x1}, {x2}), y=({y1}, {y2})")
    roi = frame[y1:y2, x1:x2]
    cv2.imwrite("debug_roi_raw.png", roi)
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.resize(roi_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, roi_thresh = cv2.threshold(roi_gray, 180, 255, cv2.THRESH_BINARY)
    cv2.imwrite("debug_roi_thresh.png", roi_thresh)
    raw_text_unrestricted = pytesseract.image_to_string(roi_thresh)
    print(f"[OCR] Raw text (tanpa restriction): {repr(raw_text_unrestricted)}")
    # OCR dengan config yang sama seperti di inference.py
    raw_text_restricted = pytesseract.image_to_string(
        roi_thresh,
        config="--psm 7 -c tessedit_char_whitelist=0123456789-:MonTueWedThuFriSatSun "
    )
    print(f"[OCR] Raw text (dengan whitelist+psm7): {repr(raw_text_restricted)}")