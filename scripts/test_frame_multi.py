"""
Alat bantu uji AD-HOC (BUKAN bagian resmi Fase 1-3 di docs/face-recognition-plan.md).

Tujuan: menguji inference/face_identity.py terhadap SATU frame CCTV yang berisi
banyak orang, tanpa menunggu integrasi penuh ke inference/new_inference.py (Fase 3).

Alurnya meniru arsitektur target plan (§4): YOLO deteksi orang -> crop per orang
-> FaceIdentifier.identify(crop) per orang -- bukan identify() langsung ke frame
penuh (itu cuma akan menguji 1 wajah terbesar, lihat diskusi sebelumnya).

Cara pakai: ganti IMAGE_PATH di bawah, lalu jalankan:
    python scripts/test_frame_multi.py
"""
import os
import sys

# scripts/ ada di dalam root project, tapi Python cuma otomatis menambahkan
# folder file ini sendiri (scripts/) ke sys.path -- bukan root -- jadi import
# lintas-folder seperti "inference.face_identity" perlu bootstrap manual ini
# supaya script tetap bisa dijalankan langsung dari mana saja.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from ultralytics import YOLO

from inference.face_identity import FaceIdentifier

# ── Ganti path ini ke frame yang mau diuji ──
IMAGE_PATH = "./frame_000044.jpg"

OUTPUT_PATH = "output/test_frame_multi_annotated.jpg"
YOLO_WEIGHTS = "yolov8l-worldv2.pt"
YOLO_CONF = 0.35


def diagnose_crop(identifier: FaceIdentifier, crop):
    """Intip detail mentah di balik identify() -- BUKAN mengubah kontrak
    FaceIdentifier, cuma mengulang logikanya di sini untuk keperluan debug,
    supaya kelihatan kenapa suatu crop gagal match (bukan cuma None polos)."""
    faces = identifier.app.get(crop)
    if not faces:
        return {"n_faces": 0}

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    x1, y1, x2, y2 = face.bbox
    face_size = min(x2 - x1, y2 - y1)

    query = face.normed_embedding.astype("float32")
    similarities = identifier.vectors @ query

    best_per_employee = {}
    for eid, sim in zip(identifier.employee_ids, similarities):
        sim = float(sim)
        if eid not in best_per_employee or sim > best_per_employee[eid]:
            best_per_employee[eid] = sim

    ranked = sorted(best_per_employee.items(), key=lambda kv: kv[1], reverse=True)
    return {"n_faces": len(faces), "face_size": face_size, "ranked": ranked}


def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"[ERROR] Frame tidak ditemukan: {IMAGE_PATH}")
        print("Edit IMAGE_PATH di scripts/test_frame_multi.py, arahkan ke frame yang mau diuji.")
        return

    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print(f"[ERROR] Gagal baca gambar: {IMAGE_PATH}")
        return

    print("Load YOLO (person detector)...")
    yolo_model = YOLO(YOLO_WEIGHTS)
    yolo_model.set_classes(["person"])

    print("Load FaceIdentifier...")
    identifier = FaceIdentifier()

    results = yolo_model(frame, conf=YOLO_CONF, verbose=False)
    boxes = [tuple(map(int, b.xyxy[0])) for b in results[0].boxes]

    print(f"\n[Test] {len(boxes)} orang terdeteksi YOLO di {IMAGE_PATH}\n")

    annotated = frame.copy()

    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            print(f"  Orang #{i} [{x1},{y1},{x2},{y2}]: crop kosong, dilewati")
            continue

        match = identifier.identify(crop)
        diag = diagnose_crop(identifier, crop)

        if match is None:
            label = "UNKNOWN"
            color = (128, 128, 128)
            if diag["n_faces"] == 0:
                print(f"  Orang #{i} [{x1},{y1},{x2},{y2}]: tidak match "
                      f"-- SCRFD tidak menemukan wajah sama sekali di crop ini")
            elif diag["face_size"] < identifier.min_face_px:
                print(f"  Orang #{i} [{x1},{y1},{x2},{y2}]: tidak match "
                      f"-- wajah terdeteksi tapi cuma {diag['face_size']:.0f}px "
                      f"(< {identifier.min_face_px}px minimum)")
            else:
                ranked_str = ", ".join(f"{eid}={score:.3f}" for eid, score in diag["ranked"])
                top1_eid, top1_score = diag["ranked"][0]
                top2_score = diag["ranked"][1][1] if len(diag["ranked"]) > 1 else -1.0
                print(f"  Orang #{i} [{x1},{y1},{x2},{y2}]: tidak match "
                      f"-- wajah {diag['face_size']:.0f}px, skor tertinggi {top1_eid}="
                      f"{top1_score:.3f} (min_score={identifier.min_score}), "
                      f"margin={top1_score - top2_score:.3f} (min_margin={identifier.min_margin}) "
                      f"| semua skor: {ranked_str}")
        else:
            label = f"{match.employee_name} ({match.score:.2f})"
            color = (0, 255, 0)
            print(f"  Orang #{i} [{x1},{y1},{x2},{y2}]: MATCH -> {match.employee_name} "
                  f"(score={match.score:.3f}, margin={match.margin:.3f})")

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"#{i} {label}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, annotated)
    print(f"\n[Test] Hasil beranotasi disimpan -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
