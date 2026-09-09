"""
Fase 0 (docs/face-recognition-plan.md): spike kelayakan face recognition,
dijalankan retroaktif setelah ada data nyata (dilewati di awal, sekarang
dikerjakan begitu ada frame RTSP + karyawan ter-enroll).

Baca semua frame .jpg di satu folder -> YOLO deteksi orang -> tiap crop
diuji ke FaceIdentifier -> rangkum statistik kelayakan secara AGREGAT
(bukan 1 frame per 1 frame seperti scripts/test_frame_multi.py).

Output:
    output/sample/feasibility_log.txt        -> 1 log lengkap (per-box + ringkasan agregat)
    output/sample/frame_result/<nama_frame>.jpg -> tiap frame input, beranotasi kotak+label

Cara pakai:
    python scripts/face_feasibility.py frames/rtsp_30s
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import argparse

import cv2
from ultralytics import YOLO

from inference.face_identity import FaceIdentifier

YOLO_WEIGHTS = "yolov8l-worldv2.pt"
YOLO_CONF = 0.35

OUTPUT_DIR = "output/sample"
FRAME_RESULT_DIR = os.path.join(OUTPUT_DIR, "frame_result")
LOG_PATH = os.path.join(OUTPUT_DIR, "feasibility_log.txt")


def diagnose_crop(identifier: FaceIdentifier, crop):
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


def pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}" if total else "0.0"


def main():
    parser = argparse.ArgumentParser(description="Spike kelayakan face recognition dari folder frame.")
    parser.add_argument("frame_dir", help="Folder berisi frame .jpg (mis. frames/rtsp_30s)")
    args = parser.parse_args()

    frame_paths = sorted(glob.glob(os.path.join(args.frame_dir, "*.jpg")))
    if not frame_paths:
        print(f"[ERROR] Tidak ada file .jpg di {args.frame_dir}")
        return

    os.makedirs(FRAME_RESULT_DIR, exist_ok=True)
    log_file = open(LOG_PATH, "w")

    def log(msg: str = ""):
        print(msg)
        log_file.write(msg + "\n")

    log("Load YOLO (person detector)...")
    yolo_model = YOLO(YOLO_WEIGHTS)
    yolo_model.set_classes(["person"])

    log("Load FaceIdentifier...")
    identifier = FaceIdentifier()

    total_boxes = 0
    n_zero_face = 0
    n_too_small = 0
    n_scored = 0
    n_matched = 0
    face_sizes = []
    top1_scores = []
    top1_scores_by_employee: dict[str, list[float]] = {}

    log(f"\n[Feasibility] Memproses {len(frame_paths)} frame dari {args.frame_dir}/\n")

    for frame_path in frame_paths:
        frame_name = os.path.basename(frame_path)
        frame = cv2.imread(frame_path)
        if frame is None:
            log(f"  [WARN] Gagal baca: {frame_path}")
            continue

        results = yolo_model(frame, conf=YOLO_CONF, verbose=False)
        boxes = [tuple(map(int, b.xyxy[0])) for b in results[0].boxes]

        log(f"-- {frame_name}: {len(boxes)} orang terdeteksi")
        annotated = frame.copy()

        for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                log(f"   #{i} [{x1},{y1},{x2},{y2}]: crop kosong, dilewati")
                continue
            total_boxes += 1

            diag = diagnose_crop(identifier, crop)
            match = identifier.identify(crop)

            if diag["n_faces"] == 0:
                n_zero_face += 1
                label, color = "UNKNOWN", (128, 128, 128)
                log(f"   #{i} [{x1},{y1},{x2},{y2}]: tidak match -- wajah tidak terdeteksi")
            else:
                face_sizes.append(diag["face_size"])

                if diag["face_size"] < identifier.min_face_px:
                    n_too_small += 1
                    label, color = "UNKNOWN", (128, 128, 128)
                    log(f"   #{i} [{x1},{y1},{x2},{y2}]: tidak match -- "
                        f"wajah {diag['face_size']:.0f}px (< {identifier.min_face_px}px)")
                else:
                    n_scored += 1
                    top1_eid, top1_score = diag["ranked"][0]
                    top1_scores.append(top1_score)
                    top1_scores_by_employee.setdefault(top1_eid, []).append(top1_score)

                    if match is not None:
                        n_matched += 1
                        label = f"{match.employee_name} ({match.score:.2f})"
                        color = (0, 255, 0)
                        log(f"   #{i} [{x1},{y1},{x2},{y2}]: MATCH -> {match.employee_name} "
                            f"(score={match.score:.3f}, margin={match.margin:.3f})")
                    else:
                        label, color = "UNKNOWN", (128, 128, 128)
                        ranked_str = ", ".join(f"{eid}={s:.3f}" for eid, s in diag["ranked"])
                        log(f"   #{i} [{x1},{y1},{x2},{y2}]: tidak match -- "
                            f"wajah {diag['face_size']:.0f}px, skor: {ranked_str}")

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"#{i} {label}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imwrite(os.path.join(FRAME_RESULT_DIR, frame_name), annotated)

    log("\n" + "=" * 60)
    log(f"HASIL SPIKE KELAYAKAN -- {len(frame_paths)} frame, {total_boxes} box orang terdeteksi")
    log("=" * 60)
    log(f"  Wajah TIDAK terdeteksi sama sekali        : {n_zero_face} ({pct(n_zero_face, total_boxes)}%)")
    log(f"  Wajah terdeteksi tapi < {identifier.min_face_px}px             : {n_too_small} ({pct(n_too_small, total_boxes)}%)")
    log(f"  Wajah layak dibandingkan (>= {identifier.min_face_px}px)     : {n_scored} ({pct(n_scored, total_boxes)}%)")
    log(f"  Berhasil MATCH (lolos threshold saat ini) : {n_matched} ({pct(n_matched, total_boxes)}%)")

    if face_sizes:
        log(f"\n  Ukuran wajah (px)  : min={min(face_sizes):.0f} max={max(face_sizes):.0f} "
            f"avg={sum(face_sizes) / len(face_sizes):.0f}")

    if top1_scores:
        log(f"  Skor top1 (semua)  : min={min(top1_scores):.3f} max={max(top1_scores):.3f} "
            f"avg={sum(top1_scores) / len(top1_scores):.3f}")
        for eid, scores in sorted(top1_scores_by_employee.items()):
            log(f"    -> {eid}: n={len(scores)} min={min(scores):.3f} max={max(scores):.3f} "
                f"avg={sum(scores) / len(scores):.3f}")

    log(f"\n  min_score saat ini={identifier.min_score}, min_margin={identifier.min_margin} "
        f"(inference/face_identity.py)")

    log(f"\n[Feasibility] Log lengkap -> {LOG_PATH}")
    log(f"[Feasibility] Frame beranotasi -> {FRAME_RESULT_DIR}/")

    log_file.close()


if __name__ == "__main__":
    main()
