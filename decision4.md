# Decision 4 — Berhenti mengiterasi model, jalankan training sebenarnya

Iterasi kelima. Dua run: **U1** (cepstral loss, T1 dari `decision3.md`) dan **U2** (uji data bersih
1.500 ucapan tanpa regularisasi — run yang belum pernah ada). Konfigurasi dasar: LSGAN, `--d-every 1`,
cache fitur, seed 1234, validasi `dev-clean` 16 ucapan/epoch, `best.pt` = val MCD terendah.

**Ringkasan satu paragraf:** cepstral loss memberi **−1,9 dB** (rekor baru 37,90 dB), menambah data
800 → 1.500 ucapan **tidak** memberi apa pun pada jumlah langkah yang sama, dan overfitting kini
**mendatar (plateau), bukan memburuk** — sehingga, sesuai kriteria yang diminta, **iterasi perbaikan
model dihentikan** dan langkah berikutnya adalah **menjalankan training sebenarnya** (§5).

---

## 1. Hasil dua run

### U1 — cepstral loss (`--lambda-cep 5.0`), 800 ucapan × 6 epoch

| epoch | train mel | **val mel L1** | **val MCD** |
|---|---|---|---|
| 1 | 1,0702 | 0,9969 | ~42,5 |
| 2 | 0,79 ± | 0,9840 | ~40,5 |
| 3 | 0,73 ± | 0,9760 | 38,424 |
| 4 | 0,6981 | 0,9630 | **37,897** ← terbaik |
| 5 | 0,6824 | 0,9727 | 38,671 |
| 6 | 0,6742 | 0,9570 | 38,555 |

### U2 — uji data bersih (1.500 ucapan × 3 epoch = 1.125 langkah, tanpa reg, tanpa cep)

| epoch (langkah) | 800 utt (R1) | **1.500 utt (U2)** | selisih pada langkah setara |
|---|---|---|---|
| ep1 (200 / 375) | 43,092 | **41,888** | U2 unggul 1,8 dB |
| ep2 (400 / 750) | 43,721 | **40,409** | R1 unggul 0,6 dB (≈ 800 langkah R1: 39,830) |
| ep3 (600 / 1.125) | 40,701 | 40,782 | seri (R1 @1.200: 40,868) |
| ep4 (800) | **39,830** | — | — |

Karena satu epoch = 375 langkah pada 1.500 ucapan tetapi 200 langkah pada 800 ucapan, perbandingan
dilakukan pada **langkah setara**, bukan pada nomor epoch.

---

## 2. Vonis per perubahan

| Perubahan | Hasil | Vonis |
|---|---|---|
| **T1 — cepstral loss `c1..c13`** | val MCD **39,830 → 37,897 dB (−1,93)**, mel L1 juga membaik (0,9769 → 0,9630) | **BERHASIL.** Ini konfirmasi diagnosis `decision3.md` §4: yang salah adalah **objektif**, bukan varians. MCD dihitung pada cepstrum; menambahkan suku cepstrum membuat training mengoptimalkan besaran yang benar-benar dinilai. |
| **T2 — content layer 6+9** | tidak dijalankan | ditunda: butuh cache kedua (~20 menit build) dan penggabungan saat baca; lihat §5.2 |
| **T3 — data 12.000** | belum dijalankan (mahal); proksi terdekat = U2 | proksi 800→1.500 **datar** → data bukan pengungkit pada skala kecil; tetap perlu diuji pada 12k, tetapi **bukan** lagi alasan untuk menunda training |

Efek kumulatif antar iterasi (val MCD, protokol sama):

| Iterasi | Perubahan kunci | Val MCD terbaik |
|---|---|---|
| 2 (A) | hinge, ckpt by train loss | 43,125 |
| 3 (S) | LSGAN + ckpt by val MCD | 39,784 |
| 4 (R1) | `--d-every 1` | 39,830 |
| **5 (U1)** | **+ cepstral loss** | **37,897** |

---

## 3. Apakah masih overfitting? — **Tidak lagi menjadi pembatas**

Bukti yang dipakai untuk memutuskan:

| Uji | Hasil | Bacaan |
|---|---|---|
| Kurva val setelah titik optimum | U1: 37,90 → 38,67 → 38,56 (**berayun di dalam derau ±0,5 dB**) | **plateau**, bukan divergensi seperti iterasi 2 (43 → 45 monoton) |
| Menambah langkah | R1: optimum 800 langkah, lalu datar | langkah bukan pengungkit |
| Menambah data (2×) | U2 ≈ R1 pada langkah setara | data pada skala ini bukan pengungkit |
| Menekan kapasitas (dropout/wd/augment) | R2 −5,2 dB, R4 −2,8 dB | kapasitas **bukan** penyebabnya |
| Memperbaiki objektif | U1 **−1,9 dB** | **ini** pengungkitnya |

Gap train–val masih ~42 % secara angka (train mel 0,674 vs val 0,957), tetapi **gap itu tidak bisa
diperbaiki dengan cara-cara anti-overfit**: tiga percobaan pengekangan semuanya memperburuk, dan
memperbaiki objektif langsung memperbaiki val. Artinya sisa gap mencerminkan **keterbatasan kelas
fungsi/objektif**, bukan penghafalan data latih. Early stopping + checkpoint berbasis val sudah
menjaga agar tidak ada kerugian praktis.

**Kesimpulan §3: overfitting terkendali dan bukan lagi pembatas utama → berhenti mengiterasi model.**

---

## 4. Konfigurasi terkunci (champion)

```
--loss lsgan  --d-every 1  --lambda-mel 10.0  --lambda-cep 5.0  --lambda-detail 2.0
--lambda-adv 1.0  --lambda-fm 2.0  --lambda-spk 0.0
--dropout 0  --weight-decay 0  --augment 0          (regulasi: terbukti merugikan)
--val-every 1  --val-utterances 24  --early-stop-patience 3   (best.pt = val MCD terendah)
content: hubert layer 9 | speaker: xtts 512-d | f0: on | segment 4s | batch 4 | lr 2e-4 cosine
```

Sudah ditanamkan ke `train-short.bat`, `train-medium.bat`, `train-full.bat`, `train-12k.bat`.

---

## 5. KEPUTUSAN: hentikan perbaikan, jalankan training sebenarnya

> **Tidak ada lagi perubahan model pada iterasi berikutnya.** Langkah berikutnya adalah
> menjalankan training pada skala korpus yang sesungguhnya dengan konfigurasi terkunci di §4.

### 5.1 Perintah eksekusi (satu perintah, sudah disiapkan)

```bat
train-12k.bat
```

Isi: (1) membangun cache 12.000 ucapan (~3,7 jam, ~17 GB, **resumable**), lalu (2) training
12.000 ucapan × 4 epoch = **12.000 langkah** (~2,8 jam pada 1,2 langkah/detik). Output:
`checkpoints_12k/best.pt`, log di `logs/cache_12k.log` dan `logs/train_12k.log`.

Setelah selesai:

```bat
eval.bat
```

### 5.2 Gerbang keberhasilan training besar (dibandingkan baseline iterasi 3)

| Ukuran | Baseline sekarang | Target setelah training 12k |
|---|---|---|
| val MCD (dev-clean) | 37,897 dB | **< 34 dB** |
| MCD evaluasi 20 ucapan speaker tak dikenal | 42,705 dB | **< 38 dB** |
| SECS target vs sumber | 0,333 vs 0,358 | margin **positif** (butuh λ_spk > 0, lihat 5.3) |
| voiced fraction (diagnostik) | 0,061 (asli 0,322) | **> 0,20** |

Jika val MCD **tidak** turun di bawah 34 dB setelah 12k langkah, barulah buka kembali perbaikan model —
urutan yang sudah disiapkan: **T2 (HuBERT layer 6+9, untuk prosodi)** → F0 channel scaling →
voicing supervision → vocoder wideband. Semua itu **ditunda**, bukan dibatalkan.

### 5.3 Satu hal yang perlu diputuskan setelah training besar, bukan sekarang

Timbre belum berpindah (SECS target < sumber). Sesuai `decision2.md` S4, `λ_spk` baru diaktifkan
**setelah** kualitas akustik stabil: training besar ini tetap `--lambda-spk 0`. Jika gerbang §5.2
terpenuhi, run lanjutan (`--resume checkpoints_12k/best.pt --lambda-spk 0.5`) menjadi langkah
berikutnya untuk menguji pemindahan timbre.

### 5.4 Biaya dan risiko yang diterima

| Risiko | Mitigasi |
|---|---|
| 3,7 jam pembangunan cache bisa terputus | `cache.py` resumable; jalankan ulang perintah yang sama |
| ~17 GB disk | cek ruang sebelum mulai; hapus `checkpoints_ablation/` + `checkpoints_it4/` (764 MB) bila perlu |
| 2,8 jam training di CPU (listrik/panas) | `--max-minutes` tersedia; checkpoint tiap epoch; early stop 3 |
| Hasil bisa mengecewakan | gerbang §5.2 eksplisit; keputusan dibuka lagi bila gagal |

---

## 6. Catatan kejujuran

1. Validasi hanya **16 ucapan**; selisih <1 dB (mis. R1 vs S = 0,05 dB) tidak boleh dianggap nyata.
   Training besar memakai **24** ucapan agar keputusan akhir lebih stabil.
2. U2 hanya 3 epoch (1.125 langkah) pada 1.500 ucapan — perbandingan langkah-setara dilakukan manual
   (`ep1`=375 langkah vs `ep1`+`ep2` R1=400 langkah), sehingga selisih <0,6 dB bersifat indikatif.
3. T2 (layer 6+9) dan T3 (12k) belum dijalankan; klaim "data bukan pengungkit" **hanya** berlaku pada
   skala 800–1.500 ucapan.
4. Gap train–val 42 % **masih ada**; yang berubah adalah penilaian bahwa menekannya tidak lagi
   menghasilkan perbaikan terukur. Ini keputusan berbasis bukti (3 run negatif), bukan klaim bahwa
   overfitting hilang.
5. Angka bersumber dari `logs/it5_U1.log`, `logs/it5_U2.log`, `logs/it4_R1.log`, `logs/abl_S.log`,
   `eval_out/report.csv`.
