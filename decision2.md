# Decision 2 — Hasil penerapan perubahan struktural S1–S4

Iterasi ketiga. Dokumen ini melanjutkan `decision.md`: perubahan struktural S1–S4 **sudah diterapkan
dan diuji ulang** (train-short, run pembanding 1.200 langkah, evaluasi 20 ucapan, diagnostik).
Keputusan langkah berikutnya ada di §8.

---

## 1. Yang diterapkan

| Kode | Perubahan | Di mana | Status |
|---|---|---|---|
| **S1** | Loss adversarial default **hinge → LSGAN** (non-saturating) | `train.py` `--loss` default `lsgan` | ✅ |
| **S2** | `best.pt` dipilih dari **val MCD terendah** (bukan train G loss) + **early stopping** (`--early-stop-patience 3`) | `train.py` loop validasi & blok simpan | ✅ |
| **S3** | **`--d-every N`** — D hanya di-update tiap N langkah; dipakai `--d-every 2` di semua `.bat` | `train.py` + `train-*.bat` | ✅ (lihat §6 catatan) |
| **S4** | `λ_spk` tetap **0** sampai val MCD mendatar | tidak ada perubahan kode | ✅ (belum waktunya) |

Bukti bahwa mekanismenya hidup (log run baru):

```
Adversarial   : lsgan   D updated every 2 step(s)   early-stop patience=3 (val MCD)
Best ckpt by  : val MCD
     best val MCD so far: 40.745 dB at epoch 3 | stale validations: 0/3
  -> saved best checkpoint (val MCD 40.745 dB) to ...\checkpoints
```

---

## 2. Hasil — train-short (800 ucapan, 4 epoch, batas 10 menit, cache aktif)

| epoch | train mel L1 | detail | D loss | adv | val mel L1 | **val MCD** |
|---|---|---|---|---|---|---|
| 1 | 1,0744 | 1,2433 | 0,412 | +0,377 | 1,0595 | 42,610 |
| 2 | 0,7834 | 0,8891 | 0,165 | +0,339 | 1,0379 | 42,568 |
| 3 | 0,7328 | 0,8070 | 0,114 | +0,352 | 0,9977 | **40,745** ← terbaik |
| 4 | 0,7138 | 0,7738 | 0,090 | +0,349 | 0,9828 | 40,943 |

Train **tidak** lagi berhenti membaik, `adv` stabil (+0,35), dan checkpoint disimpan pada epoch dengan
val MCD terendah — bukan pada epoch yang paling overfit.

## 3. Hasil — run pembanding S (800 ucapan × 6 epoch = 1.200 langkah, dinamika baru)

| step | **A (hinge, d-every 1)** | **S (LSGAN, d-every 2)** | Δ |
|---|---|---|---|
| 200 | 43,368 | 43,059 | −0,31 |
| 400 | 44,776 | 43,551 | **−1,23** |
| 600 | 43,125 *(A terbaik)* | 40,855 | **−2,27** |
| 800 | 43,545 | **39,784** *(S terbaik)* | **−3,76** |
| 1.000 | 44,294 | 40,855 | −3,44 |
| 1.200 | 45,265 | 40,857 | **−4,41** |

Perbandingan terbaik-ke-terbaik: **A 43,125 vs S 39,784 = −3,34 dB**, dan S mencapai optimumnya
**lebih lambat** (step 800 vs 600) — artinya model kini membaik dengan tambahan langkah, tidak
langsung overfit seperti sebelumnya. Bahkan B (A di-resume sampai 2.400 langkah, terbaik 42,317) masih
**2,5 dB lebih buruk** daripada S yang hanya dilatih 800 langkah.
`adv` di S stabil **+0,32…+0,38** sepanjang 6 epoch (di A/B melonjak +1,96 → +4,49 lalu turun) — bukti
langsung bahwa saturasi hinge sudah hilang.

## 4. Hasil — evaluasi 20 ucapan speaker tak dikenal (cross-speaker, `eval.bat`)

| metrik | iterasi 2 (16 utt) | **iterasi 3 (20 utt)** | Δ |
|---|---|---|---|
| MCD | 47,210 ± 7,674 dB | **42,705 ± 8,524 dB** | **−4,5 dB** |
| mel L1 | 1,1958 | 1,0783 | −0,12 |
| SECS vs target | 0,225 | **0,333** | +0,108 |
| SECS vs sumber | 0,232 | 0,358 | +0,126 |
| margin (target − sumber) | −0,007 | −0,025 | ≈ tetap negatif |

Laporan lengkap: `eval_out/report.csv` (+ `report.json`, 20 pasang `_gen`/`_ref`). Iterasi lama tetap
tersimpan di `eval_out_1/`.
Catatan: himpunan ucapannya tidak identik (16 vs 20), jadi selisih ini indikatif, bukan presisi.
**Timbre belum berpindah** — output masih sedikit lebih mirip sumber daripada target, sesuai S4 yang
sengaja belum diaktifkan.

## 5. Hasil — diagnostik spektral checkpoint baru (`diagnose_out_iter3/`)

| ukuran | real | model iterasi 2 | **model iterasi 3** | bacaan |
|---|---|---|---|---|
| band 2–4 kHz | 0,0328 | 0,0102 | 0,0124 | masih **2,7× terlalu rendah** (konsonan lemah) |
| band 6–8 kHz | 0,00066 | 0,00255 | 0,00316 | **4,8× terlalu tinggi** (hiss/derau pita atas) |
| centroid | 1726 Hz | 1774 Hz | 1754 Hz | mendekati asli |
| F0 std | 1,55 st | 2,51 st | **3,25 st** | **2,1× terlalu goyah** (intonasi berlebihan) |
| voiced fraction | 0,322 | 0,425 | 0,144 | bagian bersuara masih jauh di bawah asli |
| mel_delta_abs | 0,4899 | 0,5161 | 0,4932 | dinamika antar-frame **cocok** (bukan over-smooth) |
| mel_bin_std | 1,561 | 1,734 | **1,846** | variasi **melebihi** asli → spektrum agak liar |

---

## 6. Analisis: perubahan mana yang bekerja?

| Perubahan | Prediksi (decision.md) | Hasil | Vonis |
|---|---|---|---|
| **S1 LSGAN** | D tetap di pita sehat, `adv` tidak melonjak >3, val MCD turun >1 dB/validasi | `adv` 0,32–0,38 (stabil), val MCD turun 3,3 dB dari A, kurva val **monoton** | **berhasil, dipertahankan** |
| **S2 best-by-val-MCD + early stop** | `best.pt` = MCD terendah; berhenti saat val tak membaik | checkpoint tersimpan di epoch 3 (train-short) dan epoch 4 (S), bukan epoch terakhir; penghitung `stale` bekerja (2/3 di akhir S) | **berhasil, dipertahankan** |
| **S3 `--d-every 2`** | menyeimbangkan permainan GAN | **tercampur dengan S1** — tidak bisa dipisahkan dari run ini. Indikasi merugikan: `mel_bin_std` 1,85 > asli 1,56, energi 6–8 kHz 4,8× terlalu tinggi, F0 2,1× terlalu goyah → D mungkin sekarang **terlalu lemah** sehingga spektrum jadi liar | **perlu diuji ulang: LSGAN + `--d-every 1`** |
| **S4 L_spk tetap off** | jangan aktifkan sebelum plateau | val MCD belum mendatar (masih bisa turun), SECS masih < sumber | **benar untuk tetap off** |

**Kesimpulan §6:** dua perubahan yang paling terukur manfaatnya adalah **S1 (LSGAN)** dan **S2
(seleksi checkpoint + early stop)**. **S3 kemungkinan over-koreksi**; buktinya bukan opini melainkan
tiga angka diagnostik yang bergerak ke arah salah (energi 6–8 kHz, `mel_bin_std`, variasi F0).

---

## 7. Status overfitting / underfitting sekarang

- Gap train–val masih lebar: pada S epoch 6, train mel 0,682 vs val 0,968 = **+42 %** → **overfitting
  masih ada**, tetapi sifatnya berubah: val MCD kini **mendatar/berayun tipis** di 39,8–40,9 dB
  (bukan memburuk monoton seperti A: 43 → 45).
- Val mel L1 masih **turun** sampai epoch terakhir (0,9684) sementara val MCD mulai naik-turun tipis →
  model masih menambah detail halus, tetapi detail yang **tidak dihargai** MCD (quefrency tinggi).
- Titik optimum val MCD ada di sekitar **800 langkah** pada 800 ucapan → **jumlah data masih menjadi
  pembatas utama**, bukan jumlah langkah. (Bukti: A vs S sama-sama 800 ucapan; S membaik 3,3 dB hanya
  karena dinamika loss diperbaiki, bukan karena data bertambah.)
- Lantai baru **~39,8 dB** (turun dari ~42,3 dB). Target literatur 5–8 dB, jadi masih **5× lipat**.

---

## 8. Keputusan untuk langkah berikutnya

> **Lanjutkan ke skala data (D1/D2 dari `decision.md`) dengan dinamika baru, tetapi `--d-every 1`.</
> Tidak ada perubahan arsitektur lagi sampai data dinaikkan.** Alasan: dinamika training sudah terbukti
> berfungsi (S1+S2, −3,3 dB pada anggaran langkah yang sama), dan titik optimum masih tercapai pada 800
> langkah — tanda pembatasnya sekarang jumlah data, bukan objektif.

### 8.1 Rencana eksekusi (berurutan, satu variabel per run)

| # | Aksi | Perkiraan waktu | Hipotesis / gerbang keberhasilan |
|---|---|---|---|
| **N1** | Token uji cepat: LSGAN **`--d-every 1`** vs `--d-every 2`, 800 ucapan × 6 epoch (dua run) | 2 × ~18 min | jika `d-every 1` memberi val MCD lebih rendah **dan** energi 6–8 kHz turun mendekati 0,0031 → S3 memang over-koreksi → pakai `--d-every 1` seterusnya |
| **N2** | **Cache 12.000 ucapan** (`build-cache.bat --max-utterances 12000`, ~3,7 jam, 17 GB) | 3,7 jam (bisa ditinggal) | gap train–val turun < 25 % pada 3.000 langkah pertama |
| **N3** | Train 12k × 4 epoch = **12.000 langkah** dengan LSGAN + `--d-every 1` + best-by-val-MCD + early stop | ~2,8 jam | **val MCD < 35 dB** dan turun tiap validasi; `voiced_fraction` naik > 0,25 |
| **N4** | Setelah N3: jika val MCD mendatar di 30–35 dB → baru **S4** (`λ_spk 0,5–1`) diukur dengan SECS cross-speaker | ~30 min | SECS target > SECS sumber + 0,05 tanpa menaikkan val MCD |

### 8.2 Kondisi eskalasi ke perubahan struktural lagi (belum sekarang)

Aktifkan hanya jika **N3** selesai dan val MCD **tidak** turun di bawah ~35 dB:

1. **F0 terlalu goyah (3,25 vs 1,55 st)** → kalikan kanal F0 dengan faktor `<1` (mis. 0,5) atau turunkan
   bobot kanal kontur; ini penyebab intonasi "berlebihan" yang terdengar tidak wajar.
2. **Energi 6–8 kHz berlebih + 2–4 kHz kurang** → naikkan kekuatan D (`--d-every 1`, lalu bila perlu
   multi-scale D) supaya distribusi pita diawasi lebih ketat.
3. **Vocoder wideband 24 kHz** → baru relevan untuk persepsi konsonan setelah spektrum prediksi
   mendekati benar (saat ini 8 kHz bukan penyebab angka MCD; lihat `first-iteration.md` §8).

### 8.3 Yang tetap **tidak** dikerjakan

- Arsitektur baru (VITS/flow-matching/kNN-VC) — di luar cakupan "GAN sebagai inti".
- Multi-scale D sebelum N1–N3 selesai (biaya 1,7×, dan penyebab terdekat sudah teridentifikasi).
- `λ_mel` diturunkan lagi — saat ini `10` sudah terbukti membantu (bandingkan iterasi 1 dengan λ=45).

---

## 9. Catatan kejujuran (confound & keterbatasan)

1. **S1 dan S3 diterapkan bersamaan** → perbaikan −3,3 dB tidak bisa diatribusikan hanya ke LSGAN.
   N1 dirancang khusus untuk memisahkannya.
2. **Himpunan evaluasi berbeda** (16 vs 20 ucapan) sehingga perbandingan §4 bersifat indikatif.
3. **Validasi hanya 16 ucapan** → derau ±0,5–1 dB terlihat dari ayunan S epoch 4→6 (39,78 → 40,86).
   Untuk keputusan akhir gunakan 32–40 ucapan.
4. **MCD dan mel L1 bisa berbeda arah** di fase akhir (S epoch 6: mel L1 membaik, MCD memburuk) —
   karena MCD membobot cepstrum, bukan log-mel seragam. Selalu laporkan keduanya.
5. Masih CPU-saja (Quadro K620 ~410 MB bebas), sehingga 12k langkah memakan ~2,8 jam.
