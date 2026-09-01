import os
import csv
import logging
import datetime
import tiktoken
from collections import defaultdict
from markdown import markdown
from xhtml2pdf import pisa
from openai import OpenAI, APIError as OpenAIAPIError
from anthropic import Anthropic, APIError
from config import (
    ANTHROPIC_MODEL,
    ANTHROPIC_API_KEY,
    SUMMARY_OUTPUT_DIR,
    ACTIVE_ACTIVITIES,
    IDLE_ACTIVITIES,
    HF_TOKEN,
    MODEL
)

logger = logging.getLogger(__name__)

fallback_client = OpenAI(
   base_url= "https://router.huggingface.co/v1",
   api_key=HF_TOKEN,
)


main_client = Anthropic(
    api_key= ANTHROPIC_API_KEY,
)
# Baca Log

def load_log(log_path: str):
    rows = []
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log tidak ditemukan: {log_path}")
    
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    return rows    

def compute_zone_stats(rows) -> dict:
    total_rows = len(rows)
    if total_rows == 0:
        return {
            "total_data_terpantau": 0,
            "persentase_dalam_zona": "0.0%",
            "persentase_luar_zona": "0.0%",
        }

    in_zone_rows = sum(1 for r in rows if r["in_zone"] == "True")
    out_zone_rows = total_rows - in_zone_rows

    return {
        "total_data_terpantau": total_rows,
        "persentase_dalam_zona": f"{round((in_zone_rows / total_rows) * 100, 1)}%",
        "persentase_luar_zona": f"{round((out_zone_rows / total_rows) * 100, 1)}%",
    }

def compute_activity_time_stats(rows) -> dict:
    in_zone_rows = [r for r in rows if r["in_zone"] == "True"]
    total = len(in_zone_rows)

    if total == 0:
        return {
            "total_frame_in_zone": 0,
            "persentase_active_working_time": "0.0%",
            "persentase_idle_time": "0.0%",
        }

    active_frames = sum(1 for r in in_zone_rows if r["smoothed_activity"] in ACTIVE_ACTIVITIES)
    idle_frames = sum(1 for r in in_zone_rows if r["smoothed_activity"] in IDLE_ACTIVITIES)

    return {
        "total_frame_in_zone": total,
        "active_frames": active_frames,
        "idle_frames": idle_frames,
        "persentase_active_working_time": f"{round((active_frames / total) * 100, 1)}%",
        "persentase_idle_time": f"{round((idle_frames / total) * 100, 1)}%"
    }

def compute_operations_score(violation_stats: dict, activity_stats: dict) -> int:
    total_in_zone = violation_stats.get("total_frame_dipantau_area_kerja", 0)

    if total_in_zone == 0:
        return None
    
    compliance_pct = 100 - float(violation_stats["persentase_waktu_pelanggaran"].rstrip("%"))
    active_pct = float(activity_stats["persentase_active_working_time"].rstrip("%"))

    components = [compliance_pct, active_pct]
    score = round(sum(components) / len(components))
    return max(0, min(100, score))


def compute_violation_stats(rows) -> dict:
    """
    Hitung statistik ringkasan pelanggaran di Python (bukan di LLM),
    supaya angka yang dikirim ke LLM sudah pasti akurat dan tidak
    perlu dihitung ulang dari data mentah.
    """
    in_zone_rows = [r for r in rows if r["in_zone"] == "True"]
    total_in_zone_frames = len(in_zone_rows)
    violation_frames = sum(
        1 for r in in_zone_rows if r["is_confirmed_violation"] == "True"
    )

    violation_percentage = (
        round((violation_frames / total_in_zone_frames) * 100, 1)
        if total_in_zone_frames > 0 else 0.0
    )

    return {
        "total_frame_dipantau_area_kerja": total_in_zone_frames,
        "total_frame_pelanggaran": violation_frames,
        "persentase_waktu_pelanggaran": f"{violation_percentage}%",
    }

system_prompt = (
        "Anda adalah asisten yang membuat laporan RINGKASAN OPERASIONAL HARIAN untuk "
        "pemilik restoran/kafe (FnB Owner), berdasarkan hasil pemantauan kamera CCTV. "
        "Anda akan menerima angka-angka ringkasan yang SUDAH DIHITUNG sebelumnya "
        "(Overall Operations Score, persentase Active Working Time, Idle Time, "
        "persentase waktu di dalam/luar zona kerja, dan persentase waktu pelanggaran), "
        "serta data mentah berisi catatan aktivitas dan pelanggaran per track.\n\n"
        "PENTING SOAL ANGKA: SEMUA angka yang diberi label 'SUDAH DIHITUNG' harus "
        "digunakan APA ADANYA. JANGAN menghitung ulang, membulatkan ulang, atau "
        "mengarang angka sendiri dari membaca baris-baris data mentah.\n\n"
        "PENTING: sistem pelacakan CCTV saat ini belum bisa mengenali identitas individu "
        "secara akurat. JANGAN memberi label atau penomoran individual kepada karyawan "
        "(contoh yang DILARANG: 'Karyawan A', 'Karyawan 1'). Cukup gunakan kata "
        "'karyawan' secara umum, atau frasa seperti 'salah satu karyawan'. Jangan "
        "berasumsi jumlah karyawan berdasarkan banyaknya ID unik di data mentah.\n\n"
        "PENTING SOAL JAM KEJADIAN: setiap kali menyebutkan pelanggaran spesifik, WAJIB "
        "cantumkan jam kejadiannya jika tersedia (kolom violation_timestamp), contoh: "
        "'sekitar pukul 14:49'. Jika kosong/tidak terbaca ('-'), JANGAN mengarang jam.\n\n"
        "PENTING SOAL SKOR TIDAK TERSEDIA: Jika Overall Operations Score diberikan "
        "sebagai 'TIDAK TERSEDIA' atau mungkin 'NONE', berarti tidak ada aktivitas sama sekali "
        "atau orang yang terdeteksi pada sesi tersebut. Dalam kasus ini, tulis baris skor "
        "PERSIS sebagai 'Overall Operations Score: N/A' "
        "Tulis laporan dalam Bahasa Indonesia yang mudah dipahami pemilik usaha yang "
        "tidak familiar dengan istilah teknis. Jangan gunakan istilah seperti 'frame', "
        "'track_id', 'person_id', atau data mentah lainnya.\n\n"
        "Struktur laporan HARUS mengikuti urutan ini secara persis:\n"
        "1. Judul '## Executive Summary'.\n"
        "2. Baris skor: format persis 'Overall Operations Score: X/100' dengan X diambil "
        "apa adanya dari data. JANGAN gunakan simbol, emoji, atau indikator warna apa pun "
        "pada baris ini — cukup teks dan angka saja.\n"
        "3. Baris '### Activity Percentage' diikuti TEPAT 3 sub-bullet dengan format "
        "persis seperti ini (gunakan indentasi 2 spasi untuk sub-bullet, ini adalah "
        "SATU-SATUNYA bagian yang boleh pakai nested bullet):\n"
        "   - **Active Working Time**: X% — tambahkan komentar singkat 1 kalimat apakah "
        "ini tergolong tinggi/wajar/rendah.\n"
        "   - **Idle Time**: X% — tambahkan komentar singkat 1 kalimat.\n"
        "   - **Violation Time**: X% — tambahkan komentar singkat 1 kalimat, sebutkan "
        "jenis pelanggaran paling dominan jika ada di data.\n"
        "4. Baris '### Zone Percentage' diikuti TEPAT 2 sub-bullet dengan format yang "
        "sama (indentasi 2 spasi):\n"
        "   - **Waktu di Dalam Zona Kerja**: X% — komentar singkat 1 kalimat.\n"
        "   - **Waktu di Luar Zona Kerja**: X% — komentar singkat 1 kalimat, sebutkan "
        "jika angka ini tergolong tinggi dan perlu perhatian.\n"
        "5. Baris '### Perlu Diperhatikan' diikuti bullet list (bullet biasa, TANPA nested) "
        "berisi temuan penting: pelanggaran signifikan (dengan jam jika tersedia), atau "
        "hal lain yang perlu perhatian pemilik usaha. Jika tidak ada temuan berarti, "
        "tulis satu kalimat positif saja tanpa bullet list.\n"
        "6. Baris '### Rekomendasi' berisi 1-3 bullet (TANPA nested) saran tindak lanjut "
        "yang praktis, proporsional dengan tingkat keparahan temuan.\n\n"
        "Semua angka persentase pada poin 3 dan 4 WAJIB diambil apa adanya dari data yang "
        "diberikan (Active Working Time, Idle Time, Violation Time, waktu dalam zona, "
        "waktu luar zona) — JANGAN dihitung ulang atau dibulatkan ulang.\n\n"
        "Jangan gunakan tabel markdown, gambar, atau elemen HTML. Jangan gunakan emoji "
        "atau simbol indikator warna di bagian manapun dari laporan. Jangan gunakan "
        "heading level 4 (####) atau lebih dalam. Setiap bullet dan sub-bullet maksimal "
        "1-2 kalimat komentar, jangan buat paragraf panjang. Nested bullet HANYA boleh "
        "dipakai pada poin 3 (Activity Percentage) dan poin 4 (Zone Percentage) seperti "
        "dicontohkan — bagian lain tidak boleh pakai nested bullet. Selalu beri satu "
        "baris kosong sebelum dan sesudah heading."
    )


def compress_log(rows):
    pre_track = defaultdict(lambda: {
        "frames_in_zone": 0,
        "activities": defaultdict(int),
        "violation_frames": 0,
        "first_frame": None,
        "last_frame": None,
        "violation_timestamp": None,
    })
    
    for r in rows:
        if r["in_zone"] != "True":
            continue

        tid = r["track_id"]
        info = pre_track[tid]
        info["frames_in_zone"] += 1
        info["activities"][r["smoothed_activity"]] += 1
        if r["is_confirmed_violation"] == "True":
            info["violation_frames"] += 1

            # Simpan jam kejadian pertama kali track ini terkonfirmasi violation.
            # violation_timestamp konsisten sepanjang durasi violation track yang sama
            # (hanya di-update saat transisi COMPLIANT -> VIOLATION di inference.py),
            # jadi cukup ambil nilai pertama yang bukan "-".
            ts = r.get("violation_timestamp", "-")
            if info["violation_timestamp"] is None and ts and ts != "-":
                info["violation_timestamp"] = ts

        frame_no = int(r["frame"])
        if info["first_frame"] is None:
            info["first_frame"] = frame_no
        info["last_frame"] = frame_no

    summary_lines = []
    for tid, info in pre_track.items():
        dominant_activity = max(info["activities"], key=info["activities"].get)
        ts_text = info["violation_timestamp"] or "-"
        summary_lines.append(
            f"- Track {tid}: terlihat dari frame {info['first_frame']} sampai "
            f"{info['last_frame']} ({info['frames_in_zone']} frame di dalam zona). "
            f"Aktivitas dominan: '{dominant_activity}'. "
            f"Jumlah frame terindikasi pelanggaran: {info['violation_frames']}. "
        )

    return "\n".join(summary_lines) if summary_lines else "Tidak ada aktivitas yang tercatat di dalam zona."

def _call_claude(user_prompt):
    main_client_tokens = main_client.messages.count_tokens(
        model=ANTHROPIC_MODEL,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    estimated_tokens = main_client_tokens.input_tokens
    print(f"[SUMMARIZER] Estimasi input token: {estimated_tokens}")

    response = main_client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.3
    )

    return response.content[0].text.strip()

def _call_fallback_model(user_prompt):
    encoder = tiktoken.get_encoding("cl100k_base")
    estimated_tokens = len(encoder.encode(system_prompt + user_prompt))
    print(f"[SUMMARIZER] Estimasi input token: {estimated_tokens}")

    response = fallback_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1024,
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()

def call_llm_summary(
    compressed_text: str, 
    violation_stats: dict, 
    zone_stats: dict,
    activity_stats: dict,
    ops_score: int | None,
    video_name: str = ""
) -> str:
    stats_text = (
        f"- Overall Operations Score (SUDAH DIHITUNG, gunakan apa adanya, skala 0-100): {ops_score}\n"
        f"- Persentase Active Working Time (SUDAH DIHITUNG): {activity_stats['persentase_active_working_time']}\n"
        f"- Persentase Idle Time (SUDAH DIHITUNG): {activity_stats['persentase_idle_time']}\n"
        f"- Persentase waktu di dalam zona kerja (SUDAH DIHITUNG): {zone_stats['persentase_dalam_zona']}\n"
        f"- Persentase waktu di luar zona kerja (SUDAH DIHITUNG): {zone_stats['persentase_luar_zona']}\n"
        f"- Total frame terpantau di area kerja: {violation_stats['total_frame_dipantau_area_kerja']}\n"
        f"- Total frame terindikasi pelanggaran: {violation_stats['total_frame_pelanggaran']}\n"
        f"- Persentase waktu pelanggaran (SUDAH DIHITUNG, gunakan apa adanya): "
        f"{violation_stats['persentase_waktu_pelanggaran']}"
    )

    user_prompt = (
        f"Video: {video_name or '(tidak diketahui)'}\n\n"
        f"Statistik keseluruhan:\n{stats_text}\n\n"
        f"Log aktivitas (per track_id):\n{compressed_text}\n\n"
        "Tolong buat ringkasan pelanggaran pegawai untuk rentang waktu video ini "
        "sesuai instruksi yang sudah diberikan."
    )

    try:
        summary_text = _call_claude(user_prompt=user_prompt)
    except APIError as e:
        logger.warning(f"Claude failed ({type(e).__name__}), fallback ke KIMI")
        try:
            summary_text = _call_fallback_model(user_prompt=user_prompt)
        except OpenAIAPIError as e2:
            logger.error(f"Fallback juga gagal ({type(e2).__name__}): {e2}")
            raise

    return summary_text

def markdown_to_pdf(md_text: str, out_path: str):
    html_body = markdown(md_text, extensions=["extra"])
    html = f"""
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{ font-family: Helvetica, Arial, sans-serif; font-size: 11pt; }}
        h2 {{ color: #1a1a1a; }}
        h3 {{ color: #333; margin-top: 14px; }}
        ul {{ margin-left: 16px; }}
      </style>
    </head>
    <body>{html_body}</body>
    </html>
    """
    with open(out_path, "wb") as f:
        pisa_status = pisa.CreatePDF(html, dest=f)

    if pisa_status.err:
        raise RuntimeError(f"Gagal memuat pdf: {out_path}")

def generate_summary(log_path: str, video_name: str = "")->str:
    os.makedirs(SUMMARY_OUTPUT_DIR, exist_ok=True)

    rows = load_log(log_path)
    compressed = compress_log(rows)

    violation_stats = compute_violation_stats(rows)
    zone_stats = compute_zone_stats(rows)
    activity_stats = compute_activity_time_stats(rows)
    ops_score = compute_operations_score(violation_stats, activity_stats)

    summary_text = call_llm_summary(
        compressed, 
        violation_stats, 
        zone_stats,
        activity_stats,
        ops_score=ops_score,
        video_name=video_name)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"summary-{timestamp}.pdf"
    out_path = os.path.join(SUMMARY_OUTPUT_DIR, out_name)

    markdown_to_pdf(summary_text, out_path)

    print(f"[SUMMARIZER] Ringkasan disimpan -> {out_path}")
    return summary_text


if __name__ == "__main__":
    print(generate_summary(log_path="output/violation_log_20260803_235727.csv", video_name="Footage(2 mins).mp4"))