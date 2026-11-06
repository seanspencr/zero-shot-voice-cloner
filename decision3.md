# Decision 3 — Menyerang overfitting: hasil iterasi keempat

Iterasi keempat. Fokus: **mencari cara mengatasi overfitting** setelah `decision2.md` menyimpulkan
dinamika training sudah benar (LSGAN + best-by-val-MCD) tetapi gap train–val masih ~42 %.
Empat run dijalankan; semua memakai 4 s, batch 4, cache fitur, seed 1234, validasi `dev-clean`
16 ucapan/epoch, early-stop patience 3, `best.pt` = val MCD terendah.

---

## 1. Empat run dan hasilnya

| Run | Config | Data | Val MCD terbaik | pada epoch | Val MCD akhir | Train mel (akhir) |
|---|---|---|---|---|---|---|
| **S** (it-3) | LSGAN, d-every **2**, tanpa regularisasi | 800 | **39,784** | 4 | 40,857 | 0,682 |
| **R1** | LSGAN, d-every **1**, tanpa regularisasi | 800 | **39,830** | 4 | 40,868 | 0,688 |
| **R2** | d-every 1 + `wd 0,01` + `augment 0,1` (absolut) + `dropout 0,1` | 800 | 45,024 | 4 | 46,642 | 0,779 |
| **R3** | config R2 | **1.500** | 45,637 | 1 | 46,039 | 0,795 |
| **R4** | d-every 1 + `wd 0,001` + `augment 0,10` (**relatif**) + `dropout 0,05` | 800 | 42,581 | 4 | 43,693 | 0,771 |

Kurva val MCD per epoch (ringkas):

| epoch | S (d2) | R1 (d1) | R2 (reg kuat) | R4 (reg ringan) |
|---|---|---|---|---|
| 1 | 43,059 | 43,092 | 46,195 | 43,4±|
| 2 | 43,551 | 43,0± | 48,010 | 44,2± |
| 3 | 40,855 | 40,701 | 45,865 | 42,913 |
| 4 | **39,784** | **39,830** | **45,024** | **42,581** |
| 5 | 40,855 | 41,100 | 45,852 | 43,242 |
| 6 | 40,857 | 40,868 | 46,642 | 43,693 |

---

## 2. Temuan 1 — `--d-every` tidak berpengaruh (S3 netral)

S (D tiap 2 langkah) vs R1 (D tiap langkah): **39,784 vs 39,830 dB** → selisih 0,05 dB, jauh di bawah
derau validasi (±0,5–1 dB). Kurva per epoch pun hampir identik.

**Vonis:** dosis kekuatan D **bukan** pengungkit pada konfigurasi ini. S3 boleh dianggap netral; pakai
`--d-every 1` (paling sederhana, dan standar). Dugaan di `decision2.md` bahwa `d-every 2` "over-koreksi"
**tidak terbukti** pada val MCD.

## 3. Temuan 2 — Regularisasi **memperburuk** hasil (dua kali, dengan dosis berbeda)

- R2 (kuat): 45,024 dB → **+5,2 dB lebih buruk** dari R1.
- R4 (ringan, noise relatif): 42,581 dB → **+2,8 dB lebih buruk** dari R1.

Dan penyebab R2 terkuantisasi: noise `augment 0,1` waktu itu **absolut**, sedangkan std fitur konten
HuBERT = **0,286** → sigma 0,1 = **35 % dari sinyal**. Itu bukan regularisasi, itu perusakan input.
R4 sudah memakai noise relatif (10 % dari std) plus dropout 0,05 + wd 1e-3, dan **tetap lebih buruk**.

**Vonis:** pada skala data ini, model **bukan** kekurangan kapasitas yang perlu dikekang; setiap
pengekangan hanya mengurangi kemampuannya memetakan content+speaker → mel. **Overfitting di sini bukan
klasik variance-limited.** Ini juga menjelaskan kenapa menambah data 800 → 1.500 (R3 vs R2) tidak
menolong: yang dominan tetap regresinya, bukan datanya.

## 4. Temuan 3 — Yang "rusak" bukan varians, melainkan pemetaan & distribusi spektral

Diagnostik checkpoint terbaik (R1, `diagnose_out_it4/`) vs data asli:

| ukuran | real | it-3 (S) | **it-4 (R1)** | bacaan |
|---|---|---|---|---|
| band 2–4 kHz | 0,0328 | 0,0124 | **0,0291** | ✅ kini **mendekati asli** (sebelumnya 2,7× terlalu rendah) |
| band 6–8 kHz | 0,00066 | 0,00316 | 0,00185 | membaik (dari 4,8× → 2,8× terlalu tinggi) |
| centroid | 1726 Hz | 1754 Hz | **1686 Hz** | sangat dekat |
| F0 std | 1,55 st | 3,25 st | **3,99 st** | ❌ **2,6× terlalu goyah** (makin liar) |
| voiced fraction | 0,322 | 0,144 | **0,061** | ❌ makin jauh di bawah asli |
| mel_delta_abs | 0,4899 | 0,4932 | 0,5044 | dinamika cocok (bukan over-smooth) |
| mel_bin_std | 1,561 | 1,846 | 1,817 | variasi melebihi asli |

Jadi dengan dinamika yang benar (R1) **distribusi pita membaik jelas**, tetapi **prosodi makin tidak
terkendali** (F0 4,0 semitone vs 1,55; voiced fraction 0,061 vs 0,322). Model menghasilkan spektrum
yang makin "mirip secara pita" tetapi **pola bersuara/tak bersuara dan kontur nadanya salah** — output
cenderung berupa derau termodulasi, yang juga menjelaskan kenapa MCD mentok ~40 dB (cepstrum peka
terhadap struktur halus, bukan hanya energi per pita).

**Ini menggeser fokus iterasi berikutnya: dari "menahan overfit" ke "membenahi representasi prosodi &
objektif".**

---

## 5. Keputusan iterasi kelima

> **Hentikan percobaan regularisasi** (2 run, keduanya negatif, penyebabnya terkuantifikasi).
> **Kunci konfigurasi terbaik saat ini**: LSGAN + `--d-every 1` + tanpa dropout/augment/wd,
> `λ_mel 10`, `λ_detail 2`, F0 aktif, best-by-val-MCD + early stop.
> **Serang tiga hal berikut, satu per run**, karena semuanya menyasar pemetaan (bukan varians):

| # | Aksi | Alasan berbasis data | Gerbang sukses | Waktu |
|---|---|---|---|---|
| **T1** | **Cepstral loss** (`--lambda-cep`): L1 pada DCT log-mel `c1..c13` — objektif training disamakan dengan metrik evaluasi | MCD mentok ~40 dB sementara mel-L1 terus turun → model mengoptimalkan besaran yang **tidak** dihargai MCD (quefrency tinggi) | val MCD < 36 dB, dan turun walau mel-L1 stagnan | ~18 min/run |
| **T2** | **Content layer 6+9** (concat → Linear 1536→768) vs layer 9 sekarang | F0 terlalu goyah & voiced fraction rendah = prosodi hilang di bottleneck; literatur: 6+9 memperbaiki prosodi tanpa kehilangan fonetik | F0 std mendekati 1,5–2,0 st; voiced fraction > 0,20; val MCD tidak memburuk | ~18 min/run |
| **T3** | **Data 12.000 ucapan** (cache ~3,7 jam) dengan konfigurasi terkunci | semua percobaan pada ≤1.500 ucapan menunjukkan plateau ~40 dB; skala 12k (36 % korpus, 247 speaker) belum pernah diuji dengan dinamika yang sudah benar | gap train–val < 25 %, val MCD < 35 dB | ~2,8 jam |

Urutan eksekusi yang diusulkan: **T1 → T2 → T3** (dua yang pertama murah dan bisa langsung
menggeser lantai; T3 mahal dan sebaiknya dijalankan setelah objektif & representasi final).

### 5.1 Jika T1/T2 gagal menggeser ~38 dB

Baru pertimbangkan (dengan biaya lebih besar):
1. **F0 channel scaling**: kalikan kanal kontur dengan 0,3–0,5 atau ganti dengan representasi diskrit
   (pitch token) — menyerang langsung F0 2,6× terlalu goyah.
2. **Voicing supervision**: tambahkan prediksi kanal voiced/unvoiced eksplisit (auxiliary head) atau
   loss pada energi pita rendah, karena voiced fraction 0,061 vs 0,322 adalah penyimpangan terbesar.
3. **Vocoder wideband 24 kHz** — tetap ditunda: round-trip terbukti setia (`first-iteration.md` §8),
   jadi batas 8 kHz bukan penyebab angka MCD.
4. **Arsitektur lain** (VITS/flow/kNN-VC) hanya untuk bab pembanding.

### 5.2 Hal yang **tidak** akan diulang

| Jangan | Alasan |
|---|---|
| Menambah dropout / weight decay / augment lagi pada ≤1.500 ucapan | R2, R4: −5,2 dan −2,8 dB |
| Memakai noise augmentasi dalam satuan absolut | 0,1 vs std fitur 0,286 = 35 % sinyal (R2) |
| Menyetel `--d-every` | S vs R1: 0,05 dB (derau) |
| Menyimpulkan dari 1 validasi | ayunan 39,78 ↔ 41,10 antar epoch pada model yang sama |
| Mengejar "train loss turun" | R2 train mel paling buruk tetapi train loss paling rendah di akhir |

---

## 6. Catatan kejujuran

1. Validasi hanya **16 ucapan** → derau ±0,5–1 dB. Selisih <1 dB (mis. S vs R1) **tidak** boleh
   dianggap perbedaan nyata. Untuk keputusan final pakai 32–40 ucapan.
2. R3 mengubah **dua** hal (data 1.500 + regularisasi R2) sehingga tidak bisa memisahkan efek data;
   namun karena R2 sendiri sudah lebih buruk dari R1, arah kesimpulannya tetap aman.
3. `voiced_fraction` dan `f0_std` diukur dengan pelacak autokorelasi sederhana; angka absolutnya kasar,
   tetapi **arah perbandingan** antar model (0,061 vs 0,144 vs 0,425) konsisten dan bisa dipercaya.
4. Semua angka berasal dari `logs/it4_R{1,2,3,4}.log`, `logs/abl_S.log`, `diagnose_out_it4/`.
