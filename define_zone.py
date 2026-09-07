"""
Utility interaktif untuk menentukan ulang ZONE_POLYGON yang dipakai di
inference/inference.py dan inference/new_inference.py.

Cara pakai:
    python define_zone.py                          # ambil 1 snapshot langsung dari RTSP
    python define_zone.py --image frame_contoh.jpg  # pakai gambar frame yang sudah ada

Kontrol saat window terbuka:
    klik kiri    -> tambah titik polygon (urut sesuai klik, searah/berlawanan jarum jam)
    'u'          -> undo titik terakhir
    'r'          -> reset semua titik
    's' / Enter  -> selesai, cetak & simpan hasil polygon
    'q' / Esc    -> keluar tanpa menyimpan
"""
import os
import sys
import json
import argparse
import subprocess

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

RTSP_URL = (
    f"rtsp://{os.getenv('RTSP_USERNAME')}:{os.getenv('RTSP_PASSWORD')}"
    f"@{os.getenv('RTSP_IP')}:{os.getenv('RTSP_PORT')}/Streaming/Channels/101"
)

SNAPSHOT_PATH = "zone_snapshot.jpg"
OUTPUT_JSON = "zone_polygon.json"

points = []


def grab_snapshot_from_rtsp(out_path: str):
    """Ambil 1 frame snapshot langsung dari RTSP lewat ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-rtsp_transport", "tcp",
        "-i", RTSP_URL,
        "-frames:v", "1",
        "-q:v", "2",
        out_path,
    ]
    print("[define_zone] Mengambil snapshot dari RTSP...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0 or not os.path.exists(out_path):
        print(f"[ERROR] Gagal ambil snapshot dari RTSP:\n{result.stderr[-1000:]}")
        sys.exit(1)


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))


def render(base_frame):
    frame = base_frame.copy()

    for i, pt in enumerate(points):
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(i + 1), (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if len(points) >= 2:
        pts_arr = np.array(points, dtype=np.int32)
        cv2.polylines(frame, [pts_arr], isClosed=len(points) >= 3,
                      color=(0, 255, 0), thickness=2)

    help_text = "klik=tambah titik | u=undo | r=reset | s/Enter=selesai | q/Esc=batal"
    cv2.putText(frame, f"Titik: {len(points)}   {help_text}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Tentukan ulang ZONE_POLYGON secara interaktif dengan klik titik di atas gambar frame CCTV."
    )
    parser.add_argument("--image", type=str, default=None,
                         help="Path ke gambar frame yang mau dipakai. Kalau tidak diisi, "
                              "akan ambil 1 snapshot langsung dari RTSP.")
    args = parser.parse_args()

    if args.image:
        image_path = args.image
        if not os.path.exists(image_path):
            print(f"[ERROR] File gambar tidak ditemukan: {image_path}")
            sys.exit(1)
    else:
        image_path = SNAPSHOT_PATH
        grab_snapshot_from_rtsp(image_path)

    base_frame = cv2.imread(image_path)
    if base_frame is None:
        print(f"[ERROR] Gagal baca gambar: {image_path}")
        sys.exit(1)

    window_name = "Define Zone Polygon"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    print("[define_zone] Klik titik-titik polygon secara berurutan (searah atau berlawanan jarum jam, konsisten).")
    print("[define_zone] Tekan 's'/Enter kalau selesai, 'u' untuk undo, 'r' untuk reset, 'q' untuk batal.")

    while True:
        frame = render(base_frame)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("s"), 13):  # 's' atau Enter
            if len(points) < 3:
                print("[define_zone] Minimal butuh 3 titik untuk membentuk polygon, klik lagi.")
                continue
            break
        elif key == ord("u"):
            if points:
                points.pop()
        elif key == ord("r"):
            points.clear()
        elif key in (ord("q"), 27):  # 'q' atau Esc
            print("[define_zone] Dibatalkan, tidak ada perubahan disimpan.")
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyAllWindows()

    with open(OUTPUT_JSON, "w") as f:
        json.dump(points, f, indent=2)

    print("\n" + "=" * 50)
    print("ZONE_POLYGON baru:")
    print("=" * 50)
    print("ZONE_POLYGON = [")
    for x, y in points:
        print(f"    ({x}, {y}),")
    print("]")
    print(f"\nJuga disimpan ke: {OUTPUT_JSON}")
    print("\nSalin blok ZONE_POLYGON di atas ke inference/inference.py DAN "
          "inference/new_inference.py (keduanya punya salinan ZONE_POLYGON sendiri-sendiri).")


if __name__ == "__main__":
    main()
