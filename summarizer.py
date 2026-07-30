import os
import csv
import datetime
from collections import defaultdict
from markdown import markdown
from xhtml2pdf import pisa
#from openai import OpenAI
from anthropic import Anthropic
from config import (
    ANTHROPIC_MODEL,
    ANTHROPIC_API_KEY,
    SUMMARY_OUTPUT_DIR,
)

#client = OpenAI(
#    base_url= "https://router.huggingface.co/v1",
#    api_key=HF_TOKEN,
#)


client = Anthropic(
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

def estimate_token_usage(system_prompt: str, user_prompt: str, model: str) -> int:
    count = client.messages.count_tokens(
        model=ANTHROPIC_MODEL,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    return count.input_tokens

def call_llm_summary(compressed_text: str, stats: dict, video_name: str = "") -> str:
    system_prompt = (
        "Anda adalah asisten yang membuat laporan ringkasan PELANGGARAN karyawan untuk "
        "pemilik usaha, berdasarkan hasil pemantauan kamera CCTV. Anda akan menerima "
        "data mentah berisi catatan aktivitas yang terpantau di area kerja, termasuk "
        "jenis pelanggaran (seperti bermain HP, makan, tidur, atau bermalas-malasan "
        "saat jam kerja), jam kejadian pelanggaran tersebut (kolom violation_timestamp), "
        "dan angka ringkasan yang SUDAH DIHITUNG sebelumnya berupa persentase waktu "
        "pelanggaran dari total waktu pemantauan (field persentase_waktu_pelanggaran).\n\n"
        "PENTING SOAL PERSENTASE: gunakan angka persentase_waktu_pelanggaran yang sudah "
        "diberikan APA ADANYA dalam laporan Anda. JANGAN menghitung ulang, membulatkan "
        "ulang, atau mengarang persentase sendiri dari membaca baris-baris data mentah "
        "satu per satu — angka itu sudah dihitung secara akurat sebelum sampai ke Anda "
        "dan tidak boleh diubah. Sebutkan persentase ini di bagian Ringkasan Singkat "
        "dengan kalimat yang mudah dipahami, misalnya: 'Selama masa pemantauan, sekitar "
        "18% waktu tercatat sebagai indikasi pelanggaran.' Jika field persentase_waktu_pelanggaran "
        "tidak tersedia di data, JANGAN mengarang angka — cukup lewati penyebutan persentase.\n\n"
        "PENTING: Laporan ini HANYA berfokus pada pelanggaran. Jangan membuat ringkasan "
        "aktivitas umum karyawan yang bekerja normal/sesuai tugas — bagian itu boleh "
        "disebut sekilas sebagai konteks singkat, tapi bukan fokus utama laporan. "
        "Fokus utama dan mayoritas isi laporan harus tentang pelanggaran yang terpantau.\n\n"
        "PENTING: sistem pelacakan CCTV saat ini belum bisa mengenali identitas individu "
        "secara akurat — satu karyawan yang sama bisa saja tercatat sebagai beberapa entitas "
        "berbeda di data mentah. Karena itu, JANGAN memberi label atau penomoran individual "
        "kepada karyawan (contoh yang DILARANG: 'Karyawan A', 'Karyawan 1', 'Karyawan #2', "
        "'karyawan pertama'). Cukup gunakan kata 'karyawan' secara umum, atau frasa seperti "
        "'salah satu karyawan' atau 'beberapa karyawan' jika memang ada lebih dari satu temuan "
        "yang jelas terpisah. Jangan berasumsi jumlah karyawan berdasarkan banyaknya ID unik "
        "di data mentah, karena satu ID unik bisa jadi bukan satu karyawan yang berbeda.\n\n"
        "PENTING SOAL JAM KEJADIAN: setiap kali menyebutkan sebuah pelanggaran, WAJIB "
        "cantumkan jam kejadiannya jika tersedia di data (dari kolom violation_timestamp), "
        "contoh: 'sekitar pukul 14:49' atau 'pada 28-07-2026 pukul 14:49:27'. Jika tanggal "
        "dan jam sama-sama tersedia, cukup sebut jamnya saja kecuali laporan mencakup lebih "
        "dari satu hari (baru sebut tanggalnya juga). Jika violation_timestamp untuk suatu "
        "pelanggaran kosong/tidak terbaca (tanda '-'), JANGAN mengarang jam — cukup sebutkan "
        "pelanggarannya tanpa jam, atau gunakan frasa 'pada salah satu momen pemantauan'.\n\n"
        "Tulis laporan dalam Bahasa Indonesia yang mudah dipahami oleh pemilik usaha "
        "yang tidak familiar dengan istilah teknis. Jangan gunakan istilah seperti "
        "'frame', 'track_id', 'person_id', atau data mentah lainnya.\n\n"
        "Fokus laporan pada:\n"
        "- Ringkasan singkat: sebutkan secara singkat berapa lama total pemantauan, "
        "persentase waktu pelanggaran (jika tersedia), dan gambaran umum kondisi area "
        "kerja, tanpa membedakan individu, dan tanpa menjabarkan aktivitas normal secara "
        "panjang lebar.\n"
        "- Daftar pelanggaran: untuk setiap pelanggaran yang terpantau, jelaskan jenis "
        "pelanggarannya, jam kejadiannya (wajib jika tersedia), dan seberapa signifikan "
        "(jangan berlebihan menyimpulkan jika durasinya sangat singkat). Sebut sebagai "
        "'salah satu karyawan' atau 'terjadi pada salah satu momen', bukan dengan label individu.\n"
        "- Jika tidak ada pelanggaran berarti, sampaikan itu dengan positif dan laporan "
        "cukup singkat tanpa bagian daftar pelanggaran.\n"
        "- Tutup dengan kesimpulan singkat berisi rekomendasi praktis untuk pemilik usaha "
        "(misalnya perlu ditegur, dipantau lebih lanjut pada jam tertentu, atau tidak perlu tindakan). "
        "Jika persentase waktu pelanggaran cukup tinggi (di atas 30%), tekankan ini sebagai "
        "poin yang perlu perhatian di bagian Rekomendasi. Jika persentase sangat kecil "
        "(di bawah 5%), sampaikan dengan nada yang proporsional, tidak perlu terdengar "
        "mengkhawatirkan.\n\n"
        "Gunakan format Markdown sederhana:\n"
        "- '## Ringkasan Pelanggaran Karyawan' sebagai judul utama.\n"
        "- Bullet list ('- ') untuk poin-poin temuan pelanggaran.\n"
        "- Gunakan bold hanya untuk nama pelanggaran, jam kejadian, dan persentase waktu pelanggaran.\n"
        "- Tutup dengan '### Rekomendasi' berisi saran tindak lanjut.\n"
        "Jangan gunakan tabel markdown, gambar, atau elemen HTML. Jangan mengarang data "
        "yang tidak ada di log, termasuk jam atau persentase yang tidak tersedia.\n\n"
        "Struktur laporan HARUS mengikuti urutan ini secara persis, tanpa menambah heading lain:\n"
        "1. ### Ringkasan Singkat.\n"
        "2. ### Temuan Pelanggaran (gabungkan semua temuan tanpa membedakan individu karyawan, "
        "cantumkan jam kejadian di tiap poin).\n"
        "3. ### Rekomendasi.\n"
        "Jika tidak ada pelanggaran, bagian 'Temuan Pelanggaran' cukup berisi satu kalimat "
        "yang menyatakan tidak ada pelanggaran terdeteksi selama masa pemantauan.\n"
        "Jangan gunakan heading level 4 (####) atau lebih dalam. "
        "Jangan gunakan heading di tengah paragraf atau bullet list. "
        "Setiap bullet point maksimal 2-3 kalimat, jangan buat paragraf panjang dalam satu bullet. "
        "Jangan buat nested bullet list (bullet di dalam bullet). "
        "Selalu beri satu baris kosong sebelum dan sesudah heading. "
        "Jangan gunakan bullet bersarang --) atau numbering campur bullet."
    )

    stats_text = (
        f"- Total frame terpantau di area kerja: {stats['total_frame_dipantau_area_kerja']}\n"
        f"- Total frame terindikasi pelanggaran: {stats['total_frame_pelanggaran']}\n"
        f"- Persentase waktu pelanggaran (SUDAH DIHITUNG, gunakan apa adanya): "
        f"{stats['persentase_waktu_pelanggaran']}"
    )

    user_prompt = (
        f"Video: {video_name or '(tidak diketahui)'}\n\n"
        f"Statistik keseluruhan:\n{stats_text}\n\n"
        f"Log aktivitas (per track_id):\n{compressed_text}\n\n"
        "Tolong buat ringkasan pelanggaran pegawai untuk rentang waktu video ini "
        "sesuai instruksi yang sudah diberikan."
    )

    estimated_tokens = estimate_token_usage(system_prompt, user_prompt, ANTHROPIC_MODEL)
    print(f"[SUMMARIZER] Estimasi input token: {estimated_tokens}")

    #response = client.chat.completions.create(
    #    model=MODEL,
    #    messages=[
    #        {"role": "assistant", "content": system_prompt},
    #        {"role": "user", "content": user_prompt},
    #    ],
    #    temperature=0.6,
    #)

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.3
    )

    #return response.choices[0].message.content.strip()
    return response.content[0].text.strip()

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
    stats = compute_violation_stats(rows)
    summary_text = call_llm_summary(compressed, stats, video_name=video_name)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"summary-{timestamp}.pdf"
    out_path = os.path.join(SUMMARY_OUTPUT_DIR, out_name)

    markdown_to_pdf(summary_text, out_path)

    print(f"[SUMMARIZER] Ringkasan disimpan -> {out_path}")
    return summary_text


if __name__ == "__main__":
    print(generate_summary(log_path="output/violation_log_20260710_143740.csv", video_name="Footage(2 mins).mp4"))