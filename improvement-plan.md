# Improvement Plan — dari "zombie" ke suara yang bisa dipahami

Rencana ini diturunkan dari `first-iteration.md` (§5 penyebab, §6 kegagalan pendekatan sebelumnya).
Prinsipnya: **instrumen dulu, baru model**. Setiap langkah punya hipotesis, tindakan, cara ukur, dan
kriteria sukses — supaya tidak terulang kesalahan iterasi pertama ("loss turun, jadi pasti membaik").

Status implementasi (iterasi kedua, sudah dijalankan):

| Butir | Status | Keterangan |
|---|---|---|
| 0.1 `diagnose.py` | **selesai** | sudah dijalankan; hasilnya mengoreksi diagnosis (lihat `first-iteration.md` §8) |
| 0.2 baseline mean-mel & round-trip | **selesai** | mean-mel terbukti 12x lebih datar dari real; model tidak kolaps ke sana |
| 1.1 λ_mel 45 -> 10 + loss detail | **selesai** | `losses.py`: multi-scale mel L1 (1x/2x/4x) + global-variance loss, `--lambda-detail 2.0` |
| 1.2 F0 eksplisit | **selesai** | `features.f0_from_waveform` (autokorelasi, ~90 ms/klip) + `Generator.f0_proj` zero-init |
| 1.3 ablation layer 6/9/6+9 | belum | butuh run eksperimen, bukan kode |
| 1.4 vocoder wideband | belum | butuh unduhan model (izin) + latih ulang dari nol |
| 1.5 spectral norm | **selesai** | `PatchDiscriminator(spectral_norm=True)` default; multi-scale D belum |
| 1.6 L_spk / centering embedding | belum | tetap OFF; urutan sesuai rencana (setelah plateau) |
| 1.7 LR decay + grad clip | **selesai** | cosine 2e-4 -> 1e-5, `--grad-clip 1.0`, `--grad-accum` tersedia |
| 2.1 cache fitur | belum | pekerjaan terbesar yang tersisa; wajib untuk 10k+ langkah |
| 2.2 resume | **selesai** | `--resume <ckpt>` memuat G/D/probe + 3 optimizer |
| 3.x evaluasi & ablation | sebagian | `diagnose.py` + `preview.py` + `evaluate.py` siap dipakai |

**Perubahan prioritas akibat hasil diagnostik:** Fase 2 (skala) menjadi #1 karena output 150 langkah
terbukti *belum berstruktur* (voicing 3x terlalu rendah), bukan *terlalu halus*. Fase 1 tetap dikerjakan
karena tanpanya, setelah 10k langkah model akan mentok di over-smoothing.

---

## Ringkasan prioritas

| Fase | Isi | Waktu (CPU) | Dampak harapan |
|---|---|---|---|
| **0** | Instrumen diagnostik + baseline pembanding | 1–2 jam | **wajib**: memastikan penyebab sebelum menyetel apa pun |
| **1** | Perbaikan loss, F0, content layer, diskriminator, conditioning | 3–5 jam kerja kode | menyerang 4 penyebab utama (over-smoothing, prosodi, identitas, pita) |
| **2** | Skala training: cache fitur, resume, target 20k+ langkah | 2–3 jam kerja kode + jam mesin | keluar dari rezim "belum belajar" |
| **3** | Evaluasi terkontrol & tabel ablation | 1–2 jam | bukti untuk laporan, arah iterasi berikutnya |

Bila hanya punya ~1 hari mesin: kerjakan **0.1–0.2 → 1.1 + 1.2 → 2.1 + 2.3 → latih ~20k langkah → 3.1**.
Vocoder wideband (1.4) menyusul kalau setelah itu masih terdengar teredam.

---

## Fase 0 — Instrumentasi & baseline (sebelum menyentuh model)

Tanpa ini, semua perubahan berikutnya buta. Tujuan: bisa menjawab dengan angka **"bunyi buzz itu dari
vocoder, dari content, atau dari conditioning?"**

### 0.1 `diagnose.py` — analisis spektral berlapis

Ukur untuk tiga sinyal: (a) mel **asli**, (b) **round-trip vocoder** (mel asli → wav), (c) **output
model**.

| Ukuran | Cara | Arti |
|---|---|---|
| energi per pita (0–2, 2–4, 4–6, 6–8, 8–11 kHz) | STFT pada wav | apakah pita tinggi kosong (frikatif hilang) |
| spectral centroid & high-to-low energy ratio (HLR) | dari STFT | proksi ketajaman /s/ |
| F0 per frame + kontur | `librosa.pyin` (atau autokorelasi) | apakah intonasi datar |
| variasi antar-frame (`mean |Δmel|`) | selisih frame berurutan | ukuran **over-smoothing** (angka kecil = halus/lebur) |
| spektrum rata-rata & deviasi per bin | dibandingkan dengan mel asli | apakah output "menuju rata-rata" |
| MCD / mel-L1 | `metrics.py` yang sudah ada | pembanding literatur (target 5–8 dB) |

**Kriteria sukses:** keluar satu tabel `asli vs round-trip vs model`. Keputusan:
- round-trip **sudah** terdengar zombie → plafon di **vocoder/mel pipeline** → lompat ke 1.4 dulu.
- round-trip wajar tetapi model datar → penyebab di **generator/loss** → 1.1, 1.2, 1.6.

### 0.2 Dua baseline pembanding (murah, sangat informatif)

1. **Mean-mel baseline**: rata-ratakan mel dari ±200 ucapan (per bin), render lewat vocoder, dengarkan.
   *Karena*: hipotesis §5.4 menyatakan solusi termurah L1 = "suara rata-rata". Kalau file ini terdengar
   **mirip** output model, terbukti model baru sampai di situ.
2. **Copy-input baseline**: ukur MCD/SECS bila "konversi" = menyalin audio sumber apa adanya.
   Memberi batas bawah MCD (target harus di bawahnya) dan menjelaskan seberapa jauh model sekarang.

### 0.3 Protokol dengar yang sah

Selalu pakai `preview.py --utt-id <tetap>` untuk membandingkan dua checkpoint (kata sama, crop sama).
Simpan `*_meta.json`. Tulis skor 1–5 (kejelasan fonem, kemiripan timbre) di satu tabel.

**Kriteria sukses Fase 0:** ada dokumen `diagnosis.md` berisi tabel 0.1 + kesimpulan staging + baseline.
Estimasi: 1–2 jam (sebagian besar hanya load model).

---

## Fase 1 — Perbaikan penyebab

### 1.1 Turunkan dominasi L1, tambahkan multi-resolution STFT loss  ★ dampak tinggi

- **Hipotesis (§5.2):** L1 λ=45 → rata-rata kondisional → spektrum halus → buzz.
- **Tindakan:**
  - `--lambda-mel` **45 → 5–10**.
  - tambah **multi-resolution STFT loss** (3 resolusi, mis. n_fft 512/1024/2048, hop 50/120/240):
    `L_stft = mean over res (L_sc + L_mag)`, bobot `--lambda-stft 1.0–2.0`.
  - adversarial tetap λ=1 (hinge), feature matching λ=2.
- **Kenapa STFT loss:** bekerja di domain waktu-frekuensi bertingkat sehingga menghukum **detail lokal
  dan harmonik** yang tidak terlihat oleh L1 mel; dipakai sebagai pengganti/pelengkap mel-loss pada
  sistem modern ([Yoneyama et al., Interspeech 2022](https://www.isca-archive.org/interspeech_2022/yoneyama22_interspeech.pdf);
  argumen over-smoothing: [Kaneko 2017](https://www.isca-archive.org/interspeech_2017/kaneko17_interspeech.pdf),
  [CycleGAN-VC2](http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/publications/Kaneko2019ICASSP05_published.pdf)).
- **Ukur:** `mean |Δmel|` (dinamika) naik, MCD turun, uji dengar tidak "buzzy".
- **Risiko:** terlalu kecil λ_mel bisa merusak konten → pantau WER/MCD.
- **Estimasi kode:** 1–2 jam (loss STFT baru di `metrics.py`/`losses`).

### 1.2 Tambahkan F0 eksplisit sebagai input  ★ dampak tinggi (prosodi)

- **Hipotesis (§5.5):** tanpa jalur F0, intonasi dirata-ratakan → monoton/zombie.
- **Tindakan:** ekstrak **log-F0** per frame (`librosa.pyin`, hop = hop mel), lakukan
  **normalisasi per-utterance** (kurangi median → simpan kontur, buang level absolut agar tidak bocor
  identitas), lalu:
  - varian A: concat 1 kanal ke content → `in_dim` 769;
  - varian B: embedding kecil (Linear 1→16) lalu concat → lebih stabil.
  Frame tak bersuara diisi 0 (atau nilai tetangga) + kanal *voiced/unvoiced* 1-bit.
- **Kenapa:** encoder F0 mengompensasi hilangnya prosodi di bottleneck konten
  ([VPC 2024](https://www.voiceprivacychallenge.org/vp2024/docs/T33_____KIT-ISL_____Investigating_Voice_Conversion_Architecture_with_Different_Bottleneck_Features_for_the_Voice_Privacy_Challenge_2024.pdf));
  pitch ternormalisasi memberi kontrol prosodi eksplisit
  ([TASLP 2024](https://dl.acm.org/doi/pdf/10.1109/TASLP.2024.3439996)); AdaIN-VC lemah pada pitch
  ([arXiv 2207.06057](https://ar5iv.labs.arxiv.org/html/2207.06057)).
- **Ukur:** `F0 correlation` (prediksi vs target) dan `F0 RMSE` di nada semiton; target **corr > 0,7**.
- **Risiko:** F0 dari audio 24 kHz → 22,05 kHz, hop harus dibagi konsisten; biaya ekstraksi ~10 ms/ucapan
  (kecil dibanding HuBERT 700 ms).
- **Estimasi kode:** 2–3 jam (extractor + plumbing + metrik).

### 1.3 Ablation content layer: 6 vs 9 vs 6+9  ★ dampak sedang

- **Hipotesis (§5.5/§5.6):** layer 9 kaya fonetik tetapi miskin prosodi; layer 6 lebih kaya prosodi.
- **Tindakan:** tiga run pendek identik (langkah sama, seed sama, `preview.py` utt sama):
  `--hubert-layer 6`, `9`, dan opsi baru `6+9` (concat → Linear 1536→768).
- **Kenapa:** kombinasi 6+9 dilaporkan memperbaiki kualitas/prosodi dengan biaya fonetik kecil
  ([GenerTTS, Interspeech 2023](https://www.isca-archive.org/interspeech_2023/cong23_interspeech.pdf));
  HuBERT-Large layer 22 > 24 karena layer dalam makin "mekanis"
  ([arXiv 2409.08039](https://ar5iv.labs.arxiv.org/html/2409.08039)).
- **Ukur:** MCD + F0corr + uji dengar; pilih pemenang, jadikan default.
- **Estimasi kode:** 1 jam (flag `6+9` perlu proyeksi tambahan).

### 1.4 Ganti vocoder ke pita lebar (24 kHz) + domain mel baru  ★ dampak tinggi pada konsonan

- **Hipotesis (§5.3):** `mel_fmax=8000` membuang seluruh 8–12 kHz → frikatif hilang → "fonetik tidak jelas".
- **Tindakan:** pakai vocoder **24 kHz** (mis. Coqui vocoder 24 kHz, Vocos, atau decoder HiFi XTTS),
  lalu set `n_fft/win/hop` sesuai config vocoder itu dan `mel_fmax` ≈ 11–12 kHz. **Wajib** melatih ulang
  dari nol di domain mel baru (target mel berubah).
- **Kenapa:** kehilangan puncak spektral frikatif adalah penyebab langsung keluhan kejelasan
  ([ICASSP 2024](https://ieeexplore.ieee.org/document/10694458)); ketidakstabilan mel-cepstrum dekat
  Nyquist membuat suara "muffled"
  ([ICASSP 2012](http://dihana.cps.unizar.es/proceedings/ICASSP/2012/pdfs/0004525.pdf)); rekonstruksi /s/
  dan /z/ yang tajam butuh pita atas
  ([NAG/DAGA 2009](http://pub.dega-akustik.de/NAG_DAGA_2009/data/articles/000023.pdf)).
- **Catatan:** ini butuh **unduhan model** (saya tidak akan mengunduh tanpa izin) dan menaikkan jumlah
  frame (hop 256 @24 kHz = 93,75 fps, +9% dibanding 86,13 fps) → ~9% lebih lambat per langkah.
- **Ukur:** energi pita 8–11 kHz model vs asli (dari `diagnose.py`), HLR, uji dengar /s/.
- **Estimasi kode:** 2–4 jam + unduhan + retrain dari nol.

### 1.5 Perkuat diskriminator  ★ dampak sedang

- **Hipotesis (§5.8):** D satu skala tanpa spectral norm tidak menghukum artefak pita tinggi.
- **Tindakan:** (a) tambah `spectral_norm` pada semua Conv2d D; (b) buat **multi-scale 2 tingkat**
  (D pada resolusi penuh + D pada mel yang di-downsample 2×) lalu rata-ratakan loss-nya.
- **Kenapa:** diskriminator multi-skala berbasis *average pooling* malah menghapus HF; varian yang lebih
  baik memakai konvolusi multi-skala/DWT ([MSCGAN](https://dl.acm.org/doi/pdf/10.1145/3625383)); spectral
  norm menstabilkan D saat G masih lemah.
- **Ukur:** rasio D(real)/D(fake) mendekati seimbang (tidak 1,7/−1,9 seperti sekarang), `mean |Δmel|`.
- **Estimasi kode:** 1–2 jam.

### 1.6 Hidupkan kembali conditioning speaker — dengan urutan dan alat ukur yang benar  ★ dampak tinggi pada "timbre zombie"

- **Hipotesis (§5.4):** tanpa tekanan, generator memakai suara rata-rata.
- **Tindakan bertahap:**
  1. **Uji dulu apakah AdaIN bekerja**: dua embedding berbeda, content sama → ukur Δmel (MCD antar dua
     output). Kalau Δ ≈ 0, conditioning memang mati dan tidak ada gunanya menambah loss.
  2. **Mean-center embedding** speaker (kurangi rata-rata korpus) lalu L2-normalize ulang — memperbesar
     variasi efektif yang dilihat AdaIN.
  3. **Nyalakan `L_spk` HANYA setelah mel-L1 mendatar** (plateau), λ 0,5 → 2,0, dengan probe yang sudah
     distandardisasi per sampel. Verifikasi dengan SECS lintas-speaker, bukan dari nilai loss-nya.
  4. Opsi pengganti bila probe masih dieksploitasi: **speaker classifier loss** (klasifikasi 247 speaker)
     atau **speaker-verification feedback**; keduanya memberi tekanan identitas yang lebih sulit "ditipu".
- **Kenapa:** pemakaian identitas di decoder adalah resep yang benar (DeepVoice3/YourTTS), tetapi harus
  **dipaksa** oleh loss; loss generatif murni mendorong "hedging" ke suara rata-rata
  ([Speaker Similarity Gap](https://lacuna.tiptreesystems.com/direction/the-speaker-similarity-gap-in-speech-synthesis/txn_2b022953df224ae1ac897866085be710),
  [espnet #5301](https://github.com/espnet/espnet/issues/5301)).
- **Ukur:** SECS lintas-speaker: target harus **> sumber + 0,05**; Δmel dua speaker harus > ambang.
- **Estimasi kode:** 1–2 jam (+ uji ablation).

### 1.7 Kebersihan optimasi  ★ dampak kecil-sedang, murah

- **Tindakan:** cosine LR decay (mis. 2e-4 → 2e-5), gradient clipping `1.0`, dan `--grad-accum 2`
  (batch efektif 8 tanpa menambah RAM).
- **Kenapa:** dengan lr konstan dan gradien adversarial berderau, D mudah "menang" lalu G berhenti belajar
  (terlihat: D 0,59 vs adv naik ke 1,93 pada run terakhir).
- **Ukur:** kurva D/G seimbang, tidak ada lonjakan loss, MCD turun lebih cepat per langkah.
- **Estimasi kode:** 30 menit.

---

## Fase 2 — Skala training (penyebab §5.1)

### 2.1 Cache fitur ke disk  ★ wajib untuk CPU

- Simpan sekali per ucapan: `content [T,768]`, `speaker [512]`, `mel [80,T]` (atau wav 22,05 kHz) ke
  `datasets/cache/<content-source>/...`.
- **Kenapa:** HuBERT 700 ms + XTTS 246 ms per ucapan adalah ~85% waktu per langkah. Setelah cache, langkah
  menjadi compute-bound (~1 s/langkah) → **~3× lebih banyak langkah per jam**.
- **Ukur:** `steps/s` naik dari 0,28 → ~0,9; RAM turun (tidak perlu HuBERT di proses training).
- **Estimasi kode:** 2–3 jam; biaya sekali ekstraksi: 33.232 × 0,95 s ≈ **8,8 jam** untuk seluruh korpus
  (atau ~10 menit untuk 2.000 ucapan — mulai dari subset).

### 2.2 Dukungan resume

- Checkpoint sudah menyimpan G, D, probe, dan 3 optimizer + `args`. Tinggal `--resume <ckpt>` yang
  memuat semuanya dan melanjutkan `epoch`/`step`, plus simpan `optimizer` state (sudah ada).
- **Kenapa:** tanpa resume, setiap sesi CPU 10 menit mulai dari nol → mustahil mencapai 20k langkah.
- **Estimasi kode:** 1 jam.

### 2.3 Target skala yang realistis

| Target | Langkah | Perkiraan CPU (setelah cache, ~1 s/langkah) |
|---|---|---|
| Bukti kurva belajar | 2.000 | ~35 menit |
| **Minimum layak laporan** | 10.000 | ~3 jam |
| Layak dibandingkan literatur | 30.000–50.000 | ~10–14 jam (bisa dibagi 4–5 sesi dengan resume) |

Catatan: dengan GPU sewaan (mis. Colab/T4) 30k langkah bisa ~1–2 jam. Jika tersedia, ini opsi terbaik —
kode sudah `--device auto`.

---

## Fase 3 — Evaluasi yang bisa dipercaya

### 3.1 Metrik tambahan

| Metrik | Alat | Target |
|---|---|---|
| MCD test-clean | `evaluate.py` | **< 8 dB** |
| SECS lintas-speaker | `evaluate.py --enrollment-mode other-speaker` | target > sumber + 0,05 |
| **F0 correlation** | baru (pyin) | **> 0,7** |
| **energi pita 8–11 kHz** | `diagnose.py` | menyerupai asli (setelah 1.4) |
| WER (kejelasan) | ASR, mis. whisper-tiny (~150 MB, perlu izin unduh) | < 20% |
| MOS mini | 5 pendengar, skala 1–5 | ≥ 3 |

### 3.2 Matriks ablation (untuk laporan)

Satu baris per konfigurasi, kolom: MCD, SECS-target, SECS-sumber, F0corr, MOS, catatan dengar.

Baseline wajib ada di tabel: `copy-input`, `mean-mel`, `model-200-langkah`, lalu setiap perubahan
(1.1, 1.2, 1.3, 1.5, 1.6) sebagai satu baris.

---

## Urutan eksekusi yang diusulkan (satu per satu, bukan sekaligus)

1. **0.1 + 0.2** → `diagnosis.md`: tentukan apakah plafon di vocoder atau di generator. *(1–2 jam)*
2. **2.1 (subset 2.000 ucapan) + 2.2** → fondasi iterasi cepat. *(3–4 jam, sekali bayar)*
3. **1.1 (λ_mel turun + STFT loss)** → run 2.000 langkah, bandingkan `preview.py` + `diagnose.py`. *(~35 menit mesin)*
4. **1.2 (F0 input)** → run 2.000 langkah, lihat F0corr. *(~35 menit mesin)*
5. **1.3 (layer 6 / 9 / 6+9)** → 3 run × 2.000 langkah, pilih pemenang. *(~1,5 jam mesin)*
6. **1.5 + 1.7** → stabilkan, lalu naikkan ke **10.000–30.000 langkah**. *(3–14 jam mesin)*
7. **1.4 (vocoder wideband)** → hanya jika setelah langkah 6 frikatif masih kabur. *(perlu unduhan + retrain)*
8. **1.6 (L_spk)** → terakhir, setelah mel-L1 plateau, diukur dengan SECS. *(~35 menit mesin)*
9. **3.1/3.2** → evaluasi final + tabel ablation untuk laporan.

Setiap langkah: ubah **satu** hal, ukur, catat di tabel. Kalau tidak ada perbaikan terukur, kembalikan.

---

## Yang **tidak** boleh diulang dari iterasi pertama

| Jangan | Alasan (dari `first-iteration.md` §6) |
|---|---|
| Menyimpulkan kualitas dari nilai loss | L1 selalu bisa turun dengan cara menghaluskan spektrum |
| Menyetel hyperparameter tanpa `diagnose.py` | kita tidak bisa membedakan buzz dari vocoder vs generator |
| Membandingkan audio dari `--render-every` | kontennya acak; pakai `preview.py` dengan `--utt-id` tetap |
| Menambah data tanpa menambah **langkah** | di CPU, lebih banyak data per epoch = lebih sedikit langkah per menit |
| Menyalakan `L_spk` sejak langkah 0 | terbukti dieksploitasi probe; nyalakan setelah plateau |
| Mengejar "tidak ada error bentuk tensor" sebagai tujuan | itu syarat minimum, bukan ukuran kualitas |
| Mengganti komponen berdasarkan metrik yang tidak berhubungan dengan keluhan | margin diskriminasi speaker ≠ kualitas audio |

---

## Risiko & ketergantungan

| Risiko | Mitigasi |
|---|---|
| RAM hanya ~3,2 GB bebas | cache fitur di disk (mmap), `num_workers 0`, batch 4 + grad accumulation |
| GPU tidak memungkinkan (Quadro K620, ~410 MiB kosong) | CPU + cache, atau sewa GPU/Colab untuk run panjang |
| Vocoder wideband = unduhan baru | saya beri URL + path; tidak mengunduh tanpa izin |
| F0 ekstraksi menambah dependensi | `librosa.pyin` sudah tersedia (librosa 1.0 terpasang) |
| Waktu mesin untuk 30k langkah | resume (2.2) + kerja bertahap; atau turunkan target ke 10k |
| Referensi literatur sebagian berupa ringkasan sekunder | angka utama (MCD 5–8 dB, layer 9 *sweet spot*, F0 membantu) dikonfirmasi dari beberapa sumber independen |

---

## Definisi selesai (untuk iterasi kedua)

1. `diagnosis.md` ada dan menyebut **secara eksplisit** komponen mana yang menjadi plafon kualitas.
2. `preview.py` untuk utterance tetap menghasilkan audio yang **kata-katanya bisa dikenali** tanpa melihat transkrip.
3. MCD test-clean **< 12 dB** (antara sekarang 37–49 dB dan target ideal 8 dB) sebagai milestone antara.
4. Tabel ablation terisi minimal 4 baris (baseline, +STFT, +F0, +layer terbaik).
5. `arsitektur.md` diperbarui sesuai konfigurasi final (loss, F0 input, vocoder).
