"""
Fase 2 (docs/face-recognition-plan.md): modul pencocokan wajah -> employee_id.

Modul BERDIRI SENDIRI, dipakai nanti oleh inference/new_inference.py (Fase 3,
belum dikerjakan). Kontrak penting: FaceIdentifier.identify() TIDAK PERNAH
melempar exception -- kegagalan apa pun (wajah tidak ada, model error, dst)
dikembalikan sebagai None, supaya pipeline utama tidak pernah berhenti gara-gara
identifikasi wajah gagal (lihat plan §8, mode kegagalan).

__init__ BOLEH melempar exception (store tidak ada, model gagal load) --
pemanggil di new_inference.py yang nanti membungkusnya dengan try/except
(lihat plan §6.2), supaya kegagalan load tetap kelihatan jelas saat startup.

Sumber data: hasil scripts/enroll_faces.py
    weights/faces/employees.npz   -> vektor referensi (multi-prototype, per foto)
    weights/faces/employees.json  -> metadata (employee_id -> employee_name, active)

Uji berdiri sendiri:
    python inference/face_identity.py path/ke/foto.jpg
"""
import os
import json
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("face_identity")

DEFAULT_STORE_PATH = "weights/faces/employees.npz"
DEFAULT_MIN_SCORE = 0.30
DEFAULT_MIN_MARGIN = 0.10
DEFAULT_MIN_FACE_PX = 32


@dataclass
class FaceMatch:
    employee_id: str
    employee_name: str
    score: float
    margin: float


class FaceIdentifier:
    def __init__(self, store_path: str = DEFAULT_STORE_PATH,
                 min_score: float = DEFAULT_MIN_SCORE,
                 min_margin: float = DEFAULT_MIN_MARGIN,
                 min_face_px: int = DEFAULT_MIN_FACE_PX,
                 device: str = "cpu"):
        self.min_score = min_score
        self.min_margin = min_margin
        self.min_face_px = min_face_px

        metadata_path = os.path.join(os.path.dirname(store_path) or ".", "employees.json")

        if not os.path.exists(store_path):
            raise FileNotFoundError(
                f"Store embedding tidak ditemukan: {store_path} "
                f"(jalankan scripts/enroll_faces.py dulu)"
            )
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata karyawan tidak ditemukan: {metadata_path}")

        data = np.load(store_path)
        raw_vectors = data["vectors"]
        raw_ids = data["employee_ids"]

        with open(metadata_path) as f:
            metadata = json.load(f)

        self.employee_names = {m["employee_id"]: m["employee_name"] for m in metadata}
        active_ids = {m["employee_id"] for m in metadata if m.get("active", True)}

        keep_mask = np.array([eid in active_ids for eid in raw_ids])
        if not keep_mask.any():
            raise ValueError("Tidak ada karyawan aktif di store -- tidak ada yang bisa dicocokkan")

        self.vectors = raw_vectors[keep_mask].astype(np.float32)
        self.employee_ids = raw_ids[keep_mask]

        self.app = self._load_face_app(device)

        logger.info(
            f"FaceIdentifier siap: {len(set(self.employee_ids))} karyawan aktif, "
            f"{len(self.employee_ids)} vektor referensi (min_score={min_score}, "
            f"min_margin={min_margin})"
        )

    @staticmethod
    def _load_face_app(device: str):
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        providers = ["CPUExecutionProvider"]
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        app = FaceAnalysis(name="buffalo_l", providers=providers)
        ctx_id = 0 if providers[0] == "CUDAExecutionProvider" else -1
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        return app

    def identify(self, crop_bgr) -> FaceMatch | None:
        """TIDAK PERNAH melempar exception -- kegagalan apa pun -> None."""
        try:
            return self._identify_unsafe(crop_bgr)
        except Exception as e:
            logger.warning(f"identify() gagal, dianggap tidak match: {e}")
            return None

    def _identify_unsafe(self, crop_bgr) -> FaceMatch | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        faces = self.app.get(crop_bgr)
        if not faces:
            return None

        # Crop dari video live bisa memuat >1 wajah (orang lewat di belakang, dst) --
        # beda dengan foto enrollment yang dikurasi harus tepat 1 wajah. Di sini kita
        # ambil bbox terbesar, asumsi itu subjek utama crop orang ini.
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        x1, y1, x2, y2 = face.bbox
        face_size = min(x2 - x1, y2 - y1)
        if face_size < self.min_face_px:
            return None

        query = face.normed_embedding.astype(np.float32)

        # self.vectors sudah L2-normalized (dari enroll_faces.py), query juga --
        # dot product = cosine similarity.
        similarities = self.vectors @ query

        best_per_employee: dict[str, float] = {}
        for eid, sim in zip(self.employee_ids, similarities):
            sim = float(sim)
            if eid not in best_per_employee or sim > best_per_employee[eid]:
                best_per_employee[eid] = sim

        ranked = sorted(best_per_employee.items(), key=lambda kv: kv[1], reverse=True)
        top1_id, top1_score = ranked[0]
        top2_score = ranked[1][1] if len(ranked) > 1 else -1.0
        margin = top1_score - top2_score

        if top1_score < self.min_score or margin < self.min_margin:
            return None

        return FaceMatch(
            employee_id=top1_id,
            employee_name=self.employee_names.get(top1_id, top1_id),
            score=top1_score,
            margin=margin,
        )


if __name__ == "__main__":
    import sys
    import cv2

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) != 2:
        print(f"Cara pakai: python {sys.argv[0]} <path_gambar.jpg>")
        sys.exit(1)

    image_path = sys.argv[1]
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Gagal baca gambar: {image_path}")
        sys.exit(1)

    print("Load FaceIdentifier...")
    identifier = FaceIdentifier()

    result = identifier.identify(img)

    print()
    if result is None:
        print(f"Tidak ada match (di bawah threshold score={identifier.min_score}, "
              f"margin={identifier.min_margin})")
    else:
        print(f"Match: {result.employee_name} ({result.employee_id})")
        print(f"  score  : {result.score:.3f}")
        print(f"  margin : {result.margin:.3f}")
