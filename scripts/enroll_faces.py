"""
Fase 1 (docs/face-recognition-plan.md): enrollment wajah karyawan.

Baca foto dari employees/<nama_karyawan>/*.jpg, deteksi+embed wajahnya
(InsightFace: SCRFD untuk deteksi, ArcFace untuk embedding 512-dim), lalu
simpan sebagai "database" lokal:

    weights/faces/employees.npz    -> vektor mentah (multi-prototype, per foto)
    weights/faces/employees.json   -> metadata (employee_id, employee_name, dst)

Struktur folder input (nama folder = nama karyawan, otomatis jadi employee_name;
employee_id di-slug dari nama folder):

    employees/
      Budi Santoso/
        foto1.jpg
        foto2.jpg
      Siti Aminah/
        foto1.jpg

Cara pakai:
    python scripts/enroll_faces.py
    python scripts/enroll_faces.py --employees-dir employees --min-face-px 32
"""
import os
import re
import sys
import json
import argparse
from datetime import datetime, timezone

import cv2
import numpy as np

DEFAULT_EMPLOYEES_DIR = "employees"
STORE_DIR = "weights/faces"
VECTORS_PATH = os.path.join(STORE_DIR, "employees.npz")
METADATA_PATH = os.path.join(STORE_DIR, "employees.json")

DEFAULT_MIN_FACE_PX = 32
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "unknown"


def load_face_app():
    """Load InsightFace (SCRFD deteksi + ArcFace embedding), pakai GPU kalau ada."""
    import onnxruntime as ort
    from insightface.app import FaceAnalysis

    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")

    app = FaceAnalysis(name="buffalo_l", providers=providers)
    ctx_id = 0 if providers[0] == "CUDAExecutionProvider" else -1
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app


def embed_photo(app, image_path: str, min_face_px: int):
    """Return (embedding_512d, None) kalau valid, atau (None, alasan_ditolak)."""
    img = cv2.imread(image_path)
    if img is None:
        return None, "gagal dibaca (file rusak/bukan gambar)"

    faces = app.get(img)

    if len(faces) == 0:
        return None, "tidak ada wajah terdeteksi"
    if len(faces) > 1:
        return None, f"{len(faces)} wajah terdeteksi (harus tepat 1 wajah per foto)"

    face = faces[0]
    x1, y1, x2, y2 = face.bbox
    face_size = min(x2 - x1, y2 - y1)
    if face_size < min_face_px:
        return None, f"wajah terlalu kecil ({face_size:.0f}px < {min_face_px}px)"

    # insightface sudah mengembalikan embedding yang di-L2-normalize
    return face.normed_embedding.astype(np.float32), None


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T


def print_sanity_check(per_employee_vectors: dict):
    print("\n" + "=" * 60)
    print("SANITY CHECK - cosine similarity (untuk kalibrasi FACE_MIN_SCORE)")
    print("=" * 60)

    emp_ids = list(per_employee_vectors.keys())
    intra_scores = []
    for eid in emp_ids:
        vecs = per_employee_vectors[eid]
        if len(vecs) < 2:
            print(f"  intra  {eid:24s} hanya 1 foto valid, tidak bisa dihitung")
            continue
        sims = cosine_matrix(vecs, vecs)
        n = len(vecs)
        pairs = [sims[i, j] for i in range(n) for j in range(i + 1, n)]
        intra_scores.extend(pairs)
        print(f"  intra  {eid:24s} min={min(pairs):.3f} max={max(pairs):.3f} "
              f"avg={sum(pairs) / len(pairs):.3f} (n_pairs={len(pairs)})")

    inter_scores = []
    for i in range(len(emp_ids)):
        for j in range(i + 1, len(emp_ids)):
            sims = cosine_matrix(per_employee_vectors[emp_ids[i]], per_employee_vectors[emp_ids[j]])
            inter_scores.extend(sims.flatten().tolist())

    print()
    if intra_scores:
        print(f"  Intra-person (HARUS TINGGI) : min={min(intra_scores):.3f} "
              f"max={max(intra_scores):.3f} avg={sum(intra_scores) / len(intra_scores):.3f}")
    else:
        print("  Intra-person: tidak ada data (semua karyawan cuma punya 1 foto valid -- "
              "tambah foto lagi supaya ambang bisa dikalibrasi dari data nyata)")

    if inter_scores:
        print(f"  Inter-person (HARUS RENDAH) : min={min(inter_scores):.3f} "
              f"max={max(inter_scores):.3f} avg={sum(inter_scores) / len(inter_scores):.3f}")
    else:
        print("  Inter-person: tidak ada data (cuma ada 1 karyawan terdaftar)")

    if intra_scores and inter_scores:
        gap = min(intra_scores) - max(inter_scores)
        suggested = (min(intra_scores) + max(inter_scores)) / 2
        print(f"\n  Saran awal FACE_MIN_SCORE : ~{suggested:.3f} "
              f"(titik tengah batas-bawah intra vs batas-atas inter)")
        if gap <= 0:
            print("  [WARN] Intra dan inter-person SALING TUMPANG TINDIH -- ambang tunggal "
                  "tidak akan cukup memisahkan orang. Tambah/perbaiki kualitas foto enrollment "
                  "sebelum lanjut ke Fase 3.")
    print()


def main():
    parser = argparse.ArgumentParser(description="Enrollment wajah karyawan dari folder foto.")
    parser.add_argument("--employees-dir", default=DEFAULT_EMPLOYEES_DIR,
                         help=f"Folder input, isi subfolder per karyawan (default: {DEFAULT_EMPLOYEES_DIR})")
    parser.add_argument("--min-face-px", type=int, default=DEFAULT_MIN_FACE_PX,
                         help=f"Ukuran wajah minimum dalam piksel (default: {DEFAULT_MIN_FACE_PX})")
    args = parser.parse_args()

    if not os.path.isdir(args.employees_dir):
        print(f"[ERROR] Folder tidak ditemukan: {args.employees_dir}/")
        print("Buat strukturnya dulu, contoh:")
        print(f"  {args.employees_dir}/Budi Santoso/foto1.jpg")
        print(f"  {args.employees_dir}/Siti Aminah/foto1.jpg")
        sys.exit(1)

    person_folders = sorted(
        d for d in os.listdir(args.employees_dir)
        if os.path.isdir(os.path.join(args.employees_dir, d)) and not d.startswith(".")
    )
    if not person_folders:
        print(f"[ERROR] Tidak ada subfolder karyawan di {args.employees_dir}/")
        sys.exit(1)

    print("Load face detector + embedder (InsightFace buffalo_l)...")
    app = load_face_app()

    os.makedirs(STORE_DIR, exist_ok=True)

    all_vectors = []
    all_employee_ids = []
    metadata = []
    per_employee_vectors = {}

    print(f"\n[Enroll] {len(person_folders)} folder karyawan ditemukan di {args.employees_dir}/\n")

    for folder_name in person_folders:
        employee_name = folder_name
        employee_id = slugify(folder_name)
        folder_path = os.path.join(args.employees_dir, folder_name)

        photo_paths = sorted(
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        )

        if not photo_paths:
            print(f"[SKIP] {employee_name}: tidak ada foto (.jpg/.jpeg/.png) ditemukan")
            continue

        accepted = []
        rejected = []

        for photo_path in photo_paths:
            embedding, error = embed_photo(app, photo_path, args.min_face_px)
            if embedding is None:
                rejected.append((os.path.basename(photo_path), error))
            else:
                accepted.append(embedding)
                all_vectors.append(embedding)
                all_employee_ids.append(employee_id)

        status = "OK" if accepted else "GAGAL"
        print(f"[{status}] {employee_name} ({employee_id}): {len(accepted)}/{len(photo_paths)} foto diterima")
        for fname, reason in rejected:
            print(f"    ditolak - {fname}: {reason}")

        if not accepted:
            continue

        per_employee_vectors[employee_id] = np.stack(accepted)
        metadata.append({
            "employee_id": employee_id,
            "employee_name": employee_name,
            "num_photos": len(accepted),
            "num_rejected": len(rejected),
            "enrolled_at": datetime.now(timezone.utc).isoformat(),
            "active": True,
        })

    if not all_vectors:
        print("\n[ERROR] Tidak ada satu pun foto berhasil di-embed. Store tidak ditulis.")
        sys.exit(1)

    vectors_arr = np.stack(all_vectors).astype(np.float32)
    ids_arr = np.array(all_employee_ids, dtype="<U64")

    np.savez(VECTORS_PATH, vectors=vectors_arr, employee_ids=ids_arr)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n[Enroll] Tersimpan -> {VECTORS_PATH} ({vectors_arr.shape[0]} vektor, {vectors_arr.shape[1]} dim)")
    print(f"[Enroll] Tersimpan -> {METADATA_PATH} ({len(metadata)} karyawan)")

    print_sanity_check(per_employee_vectors)


if __name__ == "__main__":
    main()
