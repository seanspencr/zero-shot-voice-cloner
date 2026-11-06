# Decision — Overfitting vs Underfitting (keputusan: data atau model?)

Dokumen ini menjawab satu pertanyaan: **apakah hasil `train-short` bisa membaik dengan training lebih
besar, atau akan kolaps?** Jawabannya ditentukan oleh tiga run eksperimen (§2–§4), bukan perkiraan.
Keputusan ada di §6. **Belum ada tindakan yang dieksekusi** (sesuai permintaan).

Semua run memakai konfigurasi identik kecuali variabel yang diuji: HuBERT L9 + XTTS speaker, 4 s,
batch 4, cache fitur aktif, seed 1234, cosine LR 2e-4→1e-5, `λ_mel=10`, `λ_detail=2`, `λ_spk=0`,
spectral norm aktif, validasi = `dev-clean` 16 ucapan tiap epoch.

---

## 1. Cara membaca val MCD

- `MCD` dihitung pada cepstrum log-mel (`c1..c13`). Skala kasarnya linear terhadap error spektral:
  mel-L1 ≈ 1,0 nat → MCD ≈ 42–45 dB; sistem VC yang jadi ≈ **5–8 dB** (butuh error mel ≈ 0,1–0,2).
- **Pembanding "tidak belajar apa-apa"**: prediktor konstan (keluarkan mean mel dataset) ≈
  **mel L1 1,25** (dari σ mel nyata 1,561 × 0,798).
- Jadi val mel-L1 ≈ 1,03–1,08 berarti model hanya **~15 % lebih baik daripada konstanta**.

---

## 2. Run A — 800 ucapan · 6 epoch · 1.200 langkah

| epoch | step | train mel L1 | train detail | D loss | adv | **val mel L1** | **val MCD** |
|---|---|---|---|---|---|---|---|
| 1 | 200 | 1,0798 | 1,2340 | 0,593 | +1,96 | 1,0808 | **43,368** |
| 2 | 400 | 0,8363 | 0,9347 | **0,051** | +3,35 | 1,0798 | 44,776 |
| 3 | 600 | **0,8077** | 0,8905 | 0,084 | +3,47 | **1,0414** | **43,125** ← terbaik |
| 4 | 800 | 0,8095 | 0,8801 | 0,296 | +2,72 | 1,0508 | 43,545 |
| 5 | 1.000 | 0,8218 | 0,8892 | 0,461 | +1,90 | 1,0706 | 44,294 |
| 6 | 1.200 | 0,8362 | 0,9059 | 0,607 | +1,15 | 1,0773 | 45,265 |

**Bentuk kurva:** train turun lalu **naik lagi** setelah epoch 3; val mencapai titik terbaik di epoch 3
lalu memburuk **monoton**. Gap train–val di akhir: 0,8362 vs 1,0773 = **+29 %**.
→ Ini pola **overfitting**, bukan underfitting.

## 3. Run B — resume A, +6 epoch pada data yang sama (+1.200 langkah)

Mengisolasi efek **jumlah langkah** tanpa menambah data.

| epoch (lanjutan) | step total | train mel L1 | val mel L1 | val MCD |
|---|---|---|---|---|
| 1 | 1.400 | 0,8090 | **0,9990** | 42,401 |
| 2 | 1.600 | 0,7632 | 1,0154 | 44,015 |
| 3 | 1.800 | **0,7541** | 1,0173 | **42,317** ← terbaik global |
| 4 | 2.000 | 0,7648 | 1,0267 | 43,539 |
| 5 | 2.200 | 0,7666 | 1,0295 | 44,256 |
| 6 | 2.400 | 0,7764 | 1,0331 | 44,808 |

**Hasil:** menambah 1.200 langkah hanya memperbaiki val MCD **0,8 dB** (43,125 → 42,317) dan setelah itu
memburuk lagi menjadi 44,8. Train mel L1 turun ke 0,754 (rekor), tetapi val **tidak** ikut turun.
Gap akhir: 0,7764 vs 1,0331 = **+33 %** — semakin melebar.

## 4. Run C — 1.500 ucapan (220 speaker) · 3 epoch · 1.125 langkah

Mengisolasi efek **jumlah data** pada anggaran langkah yang kira-kira sama dengan A.

| epoch | step | train mel L1 | val mel L1 | val MCD |
|---|---|---|---|---|
| 1 | 375 | 0,9742 | 1,0552 | **43,099** ← terbaik |
| 2 | 750 | 0,8198 | 1,0498 | 43,555 |
| 3 | 1.125 | 0,8507 | 1,0816 | 45,132 |

**Hasil:** data 1,9× lebih banyak **tidak** memperbaiki val MCD pada anggaran langkah yang sama
(45,1 vs 45,3 milik A — selisih dalam batas derau), dan titik terbaiknya (43,099) masih **lebih buruk**
daripada B (42,317).

## 5. Perbandingan ketiga run

| Run | Data | Langkah | Val MCD terbaik | dicapai pada step | Train mel (akhir) | Val mel (akhir) | Gap |
|---|---|---|---|---|---|---|---|
| A | 800 | 1.200 | 43,125 | 600 | 0,8362 | 1,0773 | +29 % |
| **B** | 800 (resume) | 2.400 | **42,317** | 1.800 | 0,7764 | 1,0331 | +33 % |
| C | 1.500 | 1.125 | 43,099 | 375 | 0,8507 | 1,0816 | +27 % |
| — | konstanta (mean mel) | 0 | ≈ 50+ | — | — | 1,25 | — |
| — | target literatur | — | **5–8** | — | — | ~0,1–0,2 | — |

### Temuan

1. **Overfitting terkonfirmasi** pada semua run: train terus turun sementara val/val-MCD mencapai titik
   terbaik di 375–1.800 langkah lalu memburuk; gap melebar sampai ~30 %.
2. **Menambah langkah bukan pengungkit**: +1.200 langkah hanya memberi 0,8 dB, lalu merusak.
3. **Menambah data 1,9× (800 → 1.500) bukan pengungkit pada anggaran ini**: val MCD praktis sama.
4. **Ada lantai (floor) di ~42 dB** yang tidak tertembus oleh run mana pun, sementara train error terus
   turun ke 0,75. Artinya model **mampu menghafal** ucapan training tetapi **tidak** memetakan
   content+speaker → mel secara umum. Ini bukan sekadar kekurangan data pada skala 0,8k–1,5k.
5. **Temuan teknis penting — adversarial term mati (hinge saturasi):** di run A dan B, `D loss` jatuh ke
   **0,051 / 0,054** pada epoch 1–2. Pada hinge, `L_G = −E[D(fake)]` memberi gradien **nol** ketika
   `D(fake) ≤ −1`; jadi tepat pada saat generator harus belajar detail, sinyal adversarialnya hilang.
   Terlihat dari `adv` yang melonjak +1,96 → +3,35 → +4,49 lalu turun lagi ke +1,15: pertandingan GAN
   tidak stabil, dan fase "D menang" bertepatan dengan plateau train mel (epoch 2–3).
6. **Kriteria checkpoint salah sasaran:** `best.pt` dipilih berdasarkan *mean training G loss* terendah,
   bukan val MCD. Pada run B itu berarti checkpoint yang disimpan cenderung **model yang overfit**,
   bukan model yang terbaik di validasi.

---

## 6. Keputusan

> **Bukan "tambah data saja", dan bukan pula "rombak model".** Keputusan: **perbaiki dinamis training
> dulu (struktural, tanpa unduhan), lalu naikkan data — dengan urutan dan gerbang ukur yang jelas.**
> Alasan utamanya: lantai ~42 dB tidak bergeser walaupun data dinaikkan 2× dan langkah dinaikkan 2×,
> sehingga menambah data lebih dulu (mahal: 5.000 ucapan ≈ 1 jam cache) berisiko membuang waktu pada
> konfigurasi yang secara internal sudah rusak (gradien adversarial mati + seleksi checkpoint salah).

### 6.1 Struktural dulu (murah, tanpa unduhan) — 4 butir

| # | Perubahan | Alasan (berbasis data §2–§5) | Cara mengukur keberhasilan |
|---|---|---|---|
| S1 | **Hinge → LSGAN** (`--loss lsgan`) | hinge saturasi saat D menang (D loss 0,05 → gradien G = 0). LSGAN tidak saturasi, gradien tetap ada | `D loss` bertahan di 0,2–1,0; `adv` tidak melonjak >3; val MCD menurun >1 dB per validasi |
| S2 | **Seleksi `best.pt` berdasarkan val MCD**, bukan train G loss; tambah **early stopping** (mis. berhenti jika val MCD tidak membaik 2 validasi) | pada A/B titik terbaik ada di step 600–1.800, bukan di akhir; checkpoint sekarang justru menyimpan model overfit | `best.pt` = checkpoint dengan MCD terendah; dicatat di log |
| S3 | **Rasio update D/G dikendalikan** (mis. `--d-every 2`, D hanya di-update tiap 2 batch) + pertahankan spectral norm | D menang terlalu cepat (0,05) walau sudah spectral norm; memperlambat D menyeimbangkan permainan | D loss tetap di pita sehat, adv stabil, train dan val sama-sama turun |
| S4 | **`λ_spk` baru dinyalakan setelah** val MCD mendatar (bukan sekarang) | sebelumnya L_spk dini merusak audio (probe dieksploitasi); tanpa identitas pun val MCD sudah mentok, jadi ini bukan penyebab utama — tapi dibutuhkan untuk timbre di akhir | SECS(target) − SECS(source) > +0,05 tanpa penurunan val MCD |

### 6.2 Data setelah itu — 2 butir

| # | Perubahan | Alasan | Ukuran |
|---|---|---|---|
| D1 | **Cache 12.000 ucapan** (`build-cache.bat --max-utterances 12000`, ~3,7 jam, 17 GB) | overfitting terbukti pada 800–1.500; 12k = 36 % korpus, 247 speaker penuh | gap train–val < 15 %; val MCD turun < 20 dB |
| D2 | **Jadwal bertahap**: 12k × 4 epoch (12.000 langkah, ~2,8 jam) dengan S1–S3 aktif | 10k+ langkah adalah ambang minimum yang realistis (lihat `improvement-plan.md` §2.3) | val MCD harus turun tiap validasi; jika flat → eskalasi ke §6.3 |

### 6.3 Yang **tidak** dikerjakan sekarang (dan alasannya)

| Ditunda | Alasan |
|---|---|
| Vocoder wideband 24 kHz | butuh unduhan + latih ulang dari nol. MCD ~42 dB berasal dari **error prediksi spektral**, bukan batas pita; perbaikan pita memperbaiki persepsi konsonan, bukan angka MCD. Kerjakan setelah S1–S3 + D1 menunjukkan tren turun. |
| Multi-scale discriminator | menambah biaya komputasi ~1,7×; spectral norm + LSGAN + kontrol rasio D/G lebih murah dan menyasar penyebab yang terukur (D saturasi). |
| Arsitektur baru (VITS/flow-matching/kNN-VC) | di luar cakupan "GAN sebagai inti" untuk laporan; cukup dicatat sebagai pembanding di bab diskusi. |
| `L_spk` λ besar | risiko regresi audio yang sudah pernah terjadi; tunggu sampai val MCD mendatar. |

---

## 7. Gerbang keputusan setelah S1–S3 (satu run pendek, ~20 menit, 800 ucapan × 6 epoch)

| Hasil | Interpretasi | Tindakan berikutnya |
|---|---|---|
| val MCD turun **>1 dB per validasi** dan gap < 20 % | dinamika sudah benar, model masih underfit | lanjut **D1 + D2** (data 12k) |
| val MCD turun lalu membaik (pola A/B terulang) tetapi D sehat | masih overfit pada data kecil, dinamika OK | lanjut **D1** tanpa mengubah model lagi |
| val MCD tetap ~42–45 dB, D sehat, train turun | bukan masalah data/langkah → **representasi/objektif** | eskalasi ke wideband vocoder + multi-scale D, dan pertimbangkan layer HuBERT 6+9 |
| val MCD naik sejak awal / NaN | LSGAN atau rasio D/G salah setel | kembalikan `--loss hinge`, turunkan LR D, naikkan `--d-every` |

## 8. Yang sudah pasti dari investigasi ini

1. Menjalankan `train-short.bat` berulang **tidak akan** memperbaiki hasil — pada 800 ucapan, hasil
   terbaik ada di ~600–1.800 langkah, setelah itu memburuk.
2. Menaikkan data 2× saja (ke 1.500) **tidak** menembus lantai; perlu skala lompatan (≥ 5–12k) **plus**
   perbaikan dinamis training.
3. Penyebab "tidak bisa lebih baik dari 42 dB" **bukan** vocoder 8 kHz (round-trip terbukti setia ke asli,
   lihat `first-iteration.md` §8) dan **bukan** kekurangan langkah, melainkan: (a) gradien adversarial
   yang mati saat D menang, (b) seleksi checkpoint yang salah sasaran, (c) data yang terlalu sedikit
   untuk generalisasi.
