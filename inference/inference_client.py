import os
import io
import json
import zipfile
import requests
from config import KAGGLE_INFERENCE_URL

def call_inference(frame_dir: str, output_dir: str, timeout: int = 600) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for fn in sorted(os.listdir(frame_dir)):
            if fn.endswith(".jpg"):
                z.write(os.path.join(frame_dir, fn), fn)
    buf.seek(0)

    print(f"[InferenceClient] Sends {frame_dir} to {KAGGLE_INFERENCE_URL} ...")
    resp = requests.post(
        KAGGLE_INFERENCE_URL,
        files={"file": ("frames.zip", buf, "application/zip")},
        timeout=timeout,
    )
    resp.raise_for_status()

    result_zip_path = os.path.join(output_dir, "result.zip")
    with open(result_zip_path, "wb") as f:
        f.write(resp.content)
    with zipfile.ZipFile(result_zip_path) as z:
        z.extractall(output_dir)
    os.remove(result_zip_path)

    meta_path = os.path.join(output_dir, "result_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    else:
        meta = {}

    violation_log_path = meta.get("violation_log_path")
    if violation_log_path:
        violation_log_path = os.path.join(output_dir, os.path.basename(violation_log_path))

    return {
        "violation_log_path": violation_log_path,
        "annotated_video_path": meta.get("annotated_video_path"),
        "total_violations": meta.get("total_violations", 0),
        "total_detections": meta.get("total_detections", 0),
        "max_concurrent_persons": meta.get("max_concurrent_persons", 0),
        "output_dir": output_dir,
    }