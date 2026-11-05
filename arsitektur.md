# Arsitektur Sistem Zero-Shot Voice Conversion (kondisi kode saat ini)

Dokumen ini menjelaskan **komponen dan arsitektur jaringan yang benar-benar ada di kode sekarang**
(bukan rencana). Setiap komponen disertai alasan singkat mengapa dipilih/dipakai seperti itu.

Referensi implementasi: `libritts.py`, `features.py`, `models.py`, `train.py`, `vocoder.py`,
`metrics.py`, `evaluate.py`.

---

## 1. Alur data secara keseluruhan

```
    audio LibriTTS 24 kHz (4 detik, crop acak saat training / tengah saat evaluasi)
        │
        ├─► [A] HuBERT base, layer 9  ──► content  [B, T, 768]     (beku / tidak dilatih)
        │        (inti: "APA yang diucapkan")
        │
        ├─► [B] XTTS v2 speaker enc.  ──► speaker  [B, 512]        (beku / tidak dilatih)
        │        (inti: "SIAPA yang bicara" = timbre target)
        │
        └─► [C] AudioProcessor HiFi-GAN ─► mel target [B, 80, T]   (beku; sekaligus jadi target L1)

    content + speaker ──► [D] Generator (AdaIN) ──► mel prediksi [B, 80, T]
                                                        │
                          [E] Diskriminator PatchGAN ◄──┤  (real: mel target, fake: mel prediksi)
                          [F] Speaker Probe          ◄──┘  (opsional, default OFF)

    mel prediksi ──► [G] Vocoder HiFi-GAN (beku) ──► gelombang audio [B, 1, T×256]
```

Yang **dilatih** hanya: Generator ([D]), Diskriminator ([E]), dan Speaker Probe ([F] kalau diaktifkan).
HuBERT, encoder speaker, dan vocoder hanya dipakai sebagai *pengukur* yang dibekukan.

---

## 2. Bentuk tensor (kontrak antar komponen)

| Nama | Bentuk | Dihasilkan oleh | Simbol |
|---|---|---|---|
| content features | `[B, T, 768]` | HuBERT layer 9, diinterpolasi ke laju frame mel | "apa yang diucapkan" |
| speaker embedding | `[B, 512]`, L2-normalized | XTTS v2 speaker encoder | "suara siapa" |
| mel target / prediksi | `[B, 80, T]` | AudioProcessor HiFi-GAN / Generator | spektrogram |
| logits diskriminator | `[B, 1, 5, T/16]` | PatchGAN | validitas per-patch |
| embedding probe | `[B, 512]`, L2-normalized | SpeakerProbe | timbre yang terbaca dari mel |
| audio | `[B, 1, T×256]` | Vocoder | gelombang 22.05 kHz |

Nilai nyata untuk konfigurasi yang dipakai `.bat`: `T = 345` (segment 4.0 s), `B = 4`.

---

## 2b. Cache fitur (jalur cepat)

Ekstraksi (HuBERT ~700 ms + XTTS ~250 ms + F0 ~90 ms per ucapan) adalah ~85% waktu per langkah di CPU.
`cache.py` menyimpannya sekali per ucapan ke `datasets/cache/hubert-L9/<split>/<spk>/<chap>/<utt>.npz`:
`mel [80,T]` f32, `content [T,768]` **fp16**, `f0 [T,2]` f32, `speaker [512]` f32 (≈1,38 MB/ucapan).

- Pemotongan 4 detik dilakukan **saat baca**, di ruang frame, sehingga tiap epoch tetap memakai jendela
  acak yang berbeda tanpa ekstraksi ulang.
- Terukur: **0,4 -> 1,2 steps/s** (3x) dengan loss nyaris identik, karena extractor tidak lagi dipanggil.
- Fitur cache dihitung pada **seluruh utterance**, bukan pada crop: konteks HuBERT lebih panjang
  (cos 0,896 terhadap versi crop) dan embedding speaker lebih stabil.
- Jika sebuah klip belum ada di cache, dataset otomatis kembali ke ekstraksi on-the-fly.

## 3. Layer input & praproses

### 3.1 Sumber audio

- LibriTTS + LibriTTS-R, **24 kHz mono PCM16**. train-clean-100 = 33.232 ucapan / 247 speaker;
  dev-clean 5.736 / 40; test-clean 4.837 / 39. **Tidak ada speaker yang sama antar-split**, sehingga
  test-clean sah dipakai sebagai "speaker tak dikenal" (syarat zero-shot).
- **Crop 4 detik.** Training: posisi acak (berganti tiap epoch, seed tetap). Evaluasi/preview: crop
  tengah supaya deterministik.
  *Karena*: melatih dengan potongan panjang bervariasi membuat batch tidak bisa di-stack; crop tetap
  membuat bentuk tensor seragam sekaligus menghemat memori CPU.
- **Filter `--min-seconds` (= panjang segment)**: klip < 4 s dilewati, bukan di-*zero-pad*.
  *Karena*: 46% klip LibriTTS lebih pendek dari 4 s; kalau di-pad, hampir separuh batch berisi
  keheningan dan generator belajar "diam" sebagai solusi aman.

### 3.2 Mel target (`[B, 80, T]`)

Dihitung oleh **AudioProcessor milik vocoder HiFi-GAN yang terpasang**: `sr=22050` (resample dari
24 kHz), `n_fft=1024`, `win_length=1024`, `hop_length=256`, `n_mels=80`, `mel_fmax=8000`, log natural,
`spec_gain=1.0`, `signal_norm=false` (tanpa normalisasi rentang).
*Karena*: target training **wajib** memakai jalur praproses yang sama dengan vocoder. Kalau berbeda
(n_fft/hop/skala log), audio hasil konversi akan rusak walaupun bentuk tensornya benar.

> Konsekuensi yang perlu diketahui: `mel_fmax=8000` berarti **lebar pita dibatasi 8 kHz** — suara akan
> terdengar agak "muffled". Ini batas dari model vocoder, bukan dari GAN-nya.

### 3.3 Encoder konten — HuBERT (beku)

- `facebook/hubert-base-ls960`, diambil **layer ke-9** (768-d), bobot `.bin` 377,6 MB dari folder lokal
  `pretrained/hubert-base-ls960/`.
- Input 16 kHz (audionya di-resample dari 24 kHz).
- HuBERT menghasilkan ~50 frame/detik, sedangkan mel 86,1 frame/detik → hasil HuBERT
  **diinterpolasi linear** ke jumlah frame mel, sehingga `T` content == `T` mel.
  *Karena*: generator mengeluarkan satu frame mel per frame content; kalau laju berbeda, loss L1 tidak
  bisa dibandingkan per frame.
- Alternatif tanpa bobot: `--content-source mel-projection` (proyeksi deterministik dari mel asli) untuk
  uji pipa tanpa model.

### 3.4 Encoder speaker — XTTS v2 (beku)

- Kelas `ResNetSpeakerEncoder` (ResNet + SE + ASP pooling) milik XTTS v2, diambil dari checkpoint
  `model.pth` (1,78 GB) bagian `hifigan_decoder.speaker_encoder.*` saja (~32 MB) memakai
  `torch.load(..., mmap=True)` → hemat RAM.
- Input 16 kHz, mel internal 64 bin (`n_fft=512`, `win=400`, `hop=160`, preemphasis 0.97).
  Konfigurasi ini **harus identik** dengan cara XTTS melatihnya, kalau tidak embedding-nya tidak bermakna.
- Output **512-d, L2-normalized**.
  *Karena*: encoder ini dilatih **bersama** generator XTTS, jadi vektornya memang "bahasa instruksi"
  untuk sebuah generator suara; dan karena TTS depan-mu juga XTTS, tahap TTS dan tahap VC memakai ruang
  suara yang sama. Hasil ukur: cosine speaker sama 0,54 vs beda 0,24 (margin +0,30; resemblyzer hanya
  +0,15, placeholder +0,04).
- Alternatif tanpa bobot: `--speaker-source mel-stats` (statistik mel; lemah, hanya untuk uji pipa).

---

## 4. Generator (decoder mel) — `models.Generator`

Ini **decoder**-nya: rangkaian konvolusi 1-D yang mengubah content menjadi mel, dengan timbre
disuntikkan lewat AdaIN.

| Tahap | Operasi | Bentuk |
|---|---|---|
| 0 | *(opsional)* `f0_proj: Conv1d(2 → 256, k=3, p=1)` dijumlahkan ke hasil `conv_in`; bobot di-init **nol** sehingga mulanya tidak berpengaruh | `[B, 256, T]` |
| 1 | `Conv1d(768 → 256, kernel 7, padding 3)` + `AdaIN1d(256)` + `LeakyReLU(0.2)` | `[B, 256, T]` |
| 2 | **4 × `AdaINResBlock(256)`**: `Conv1d(256→256, k=3, p=1)` → AdaIN → LeakyReLU → `Conv1d` → AdaIN → `+ skip` → LeakyReLU | `[B, 256, T]` |
| 3 | `Conv1d(256 → 80, kernel 7, padding 3)` — **tanpa aktivasi (linear)** | `[B, 80, T]` |

Parameter: **5.458.768** (dengan speaker 512-d; 4.279.120 kalau 256-d).

**Semua stride = 1, padding "same"** → `T` terjaga persis dari content ke mel.
*Karena*: L1 loss membandingkan `[B,80,T]` dengan `[B,80,T]`; tanpa operasi yang mengubah panjang,
kesalahan bentuk tensor tidak mungkin terjadi (bukan kebetulan).
**Output linear tanpa aktivasi** *karena* targetnya log-mel mentah (nilai ±(−18…+6), bisa negatif);
`tanh`/`sigmoid` akan memotong rentang yang sah.

### 4.1 AdaIN — mekanisme penyuntik timbre

```
y = (1 + scale(z)) · (x − mean_t(x)) / (std_t(x) + ε) + bias(z)      z = speaker embedding
[scale, bias] = Linear(512 → 2·C)(z)
```

- Statistik instance dihitung **manual** (mean/std sepanjang waktu) *karena* `nn.InstanceNorm1d` gagal
  untuk `T=1` atau `batch=1`.
- Bobot `Linear` diinisialisasi **nol** → di awal training blok ini identik dengan instance norm biasa,
  lalu perlahan belajar memodulasi per channel sesuai speaker. *Karena*: mencegah guncangan besar di
  awal training ketika embedding masih "acak" bagi generator.
- Intuisi: normalisasi instance **membuang** informasi timbre dari content (level/nada per channel),
  lalu `scale`/`bias` yang bergantung speaker **mengembalikan** timbre target. Content menentukan
  *struktur temporal* (apa yang diucapkan), embedding menentukan *warna suara*.

### 4.2 Stream F0 (prosodi) — ditambahkan pada iterasi kedua

Kanal input tambahan `[B, T, 2]`: **log-F0 ternormalisasi** (deviasi dari median ucapan, semitone, diskalakan
ke [-1,1]) + **flag voiced**. Diekstrak dengan autokorelasi per frame (`features.f0_from_waveform`,
~90 ms/klip 4 s), bukan `pyin` yang 5-10x lebih lambat.

*Karena*: normalisasi instance di AdaIN memang membuang statistik per-utterance — termasuk kontur nada —
dan HuBERT layer 9 makin miskin prosodi. Tanpa kanal eksplisit, intonasi ikut dirata-ratakan sehingga
terdengar monoton ("zombie"). Normalisasi terhadap median (bukan nilai absolut) penting supaya kanal ini
membawa **kontur**, bukan identitas pembicara.

### 4.3 Fungsi aktivasi yang dipakai

| Aktivasi | Di mana | Alasan |
|---|---|---|
| `LeakyReLU(0.2)` | semua blok Generator, Diskriminator, Speaker Probe | gradien tidak mati untuk input negatif (penting di GAN, `ReLU` biasa bisa "dying") |
| **Linear** (tanpa aktivasi) | output Generator (`conv_out`) | target adalah log-mel mentah, boleh negatif |
| **Tanpa sigmoid** | output Diskriminator | loss hinge/LSGAN bekerja pada **logits mentah**; sigmoid akan menyempitkan gradien |
| **L2-normalize** | embedding speaker, output Speaker Probe | menyamakan skala semua speaker; cosine similarity jadi bermakna |
| Instance/mean-std (bukan aktivasi, tapi operasi norm) | AdaIN (manual), `InstanceNorm2d` di D, standardisasi per-sampel di Probe | menstabilkan skala lintas ucapan/speaker |

---

## 5. Diskriminator — `models.PatchDiscriminator` (PatchGAN)

Input mel diperlakukan sebagai "gambar" 1 kanal `[B, 1, 80, T]`.

| Tahap | Operasi | Bentuk |
|---|---|---|
| 1 | `Conv2d(1 → 64, k=4, s=2, p=1)` + `LeakyReLU(0.2)` | `[B, 64, 40, T/2]` |
| 2–4 | `Conv2d(k=4,s=2,p=1)` + `InstanceNorm2d` + `LeakyReLU(0.2)`, channel 64→128→256→512 | `[B, 512, 5, T/16]` |
| 5 | `Conv2d(512 → 1, k=3, s=1, p=1)` → logits | `[B, 1, 5, T/16]` |

Parameter: **2.760.001**.

- **PatchGAN** (bukan satu skor global) *karena* keputusan per-patch memaksa generator menjaga detail
  spektral **lokal** (harmonik, transient); penilaian global saja mudah "ditipu" dengan statistik kasar.
- 4 tahap `stride 2` → panjang keluaran dibagi 2^4 = **16** (`floor(L/2)` per tahap), jadi `T` ganjil pun aman;
  contoh nyata: `T=345` → logits `[B, 1, 5, 21]`.
- `InstanceNorm2d` *karena* statistik per-sampel lebih cocok daripada BatchNorm ketika batch kecil (4).
- `forward_features()` juga mengembalikan 4 peta aktivasi fitur → dipakai **feature matching loss**.

---

## 6. Speaker Probe (opsional, default **OFF**) — `models.SpeakerProbe`

| Tahap | Operasi | Bentuk |
|---|---|---|
| 0 | standardisasi **per sampel**: `(mel − mean)/std` (skalar atas semua bin & frame) | `[B, 80, T]` |
| 1–3 | 3 × `Conv1d(80→128 / 128→128 / 128→128, k=5, p=2)` + `LeakyReLU(0.2)` | `[B, 128, T]` |
| 4 | mean-pooling + std-pooling sepanjang waktu → `Linear(256 → 512)` → **L2-normalize** | `[B, 512]` |

Parameter: **347.008**. Dilatih **hanya dengan mel asli**, lalu dipakai dalam keadaan beku pada mel
buatan generator.

*Karena*: loss adversarial, L1, dan feature-matching tidak pernah menyebut "suara siapa". Tanpa probe,
solusi termurah bagi generator adalah **menyalin timbre input** (terbukti: SECS ke target 0,507 < SECS
ke sumber 0,558).

> **Default sekarang `--lambda-spk 0` (mati).** Alasannya empiris: ketika diaktifkan sejak awal, probe
> yang masih lemah justru "dieksploitasi" generator (pola yang menyenangkan probe, bukan suara wajar),
> dan audio jadi lebih buruk. Nyalakan (`--lambda-spk 0.5`…`2.0`) setelah mel-L1 mendatar.

---

## 7. Vocoder (mel → gelombang) — `vocoder.HiFiGANVocoder`

- Coqui **HiFi-GAN v2** `en/ljspeech/hifigan_v2`, **beku**: 928.514 parameter generator (128 channel),
  keluaran 22,05 kHz, `hop_length=256`, `mel_fmax=8000` → audio `[B, 1, T×256]`.
- Dipakai untuk **dua** hal: (a) menghitung mel target, (b) merender audio untuk didengarkan.
- Tidak pernah dilatih di sini. *Karena*: melatih vocoder butuh data & waktu jauh lebih besar
  (GPU); memakai vocoder pra-latih juga menjamin target mel dan suara akhir selalu konsisten.

---

## 8. Loss functions

Total loss generator:

```
L_G = λ_adv · L_adv + λ_mel · L_L1 + λ_detail · L_detail + λ_fm · L_fm + λ_spk · L_spk
```

(Diperbarui pada iterasi kedua: `λ_mel` 45 -> **10**, ditambah **L_detail** = multi-scale mel L1 +
global-variance loss, dan `L_spk` default **0**.)

| Loss | Rumus | Bobot default | Mengapa dipakai |
|---|---|---|---|
| **L_adv (G)** — adversarial | `hinge`: `−E[D(fake)]` · `lsgan`: `0.5·E[(D(fake)−1)²]` | `λ_adv = 1.0` | Satu-satunya sinyal "terdengar seperti suara asli". Tanpa ini generator hanya mengejar rata-rata L1 → hasil buram. |
| **L_D** — adversarial (diskriminator) | `hinge`: `E relu(1−D(real)) + E relu(1+D(fake))` · `lsgan`: `0.5·E[(D(real)−1)²] + 0.5·E[D(fake)²]` | — (hanya untuk D) | Melatih D memberi skor tinggi untuk mel asli dan rendah untuk mel buatan. **Hinge** dipilih sebagai default karena lebih tahan *collapse* di fase awal; **LSGAN** disediakan untuk fase lanjut saat output sudah mirip suara dan tinggal dihaluskan. |
| **L_L1** — rekonstruksi mel | `E‖mel_pred − mel_target‖₁` | `λ_mel = 10.0` (dulu 45) | **Karena** ini satu-satunya loss yang memberi tahu generator "mel yang benar itu seperti apa" per frame; adversarial saja tidak punya target. Namun bobot 45 berasal dari resep **vocoder**; untuk converter ia membuat generator mengejar rata-rata kondisional (blur). Iterasi kedua menurunkannya ke 10 dan memindahkan tekanannya ke L_detail. |
| **L_fm** — feature matching | `(1/L) Σ_l E‖D_l(fake) − D_l(real)‖₁` (L = 4 peta fitur D) | `λ_fm = 2.0` | **Karena** ia membandingkan *aktivasi diskriminator* untuk mel asli vs buatan: generator diberi target antara yang jauh lebih halus daripada sinyal biner real/fake, sehingga training GAN lebih stabil dan tekstur tidak "lompat-lompat". |
| **L_detail** — anti-over-smoothing | `multi_scale_mel_loss(1x,2x,4x) + GV` ; `GV = E‖std_t(pred) − std_t(target)‖₁` | `λ_detail = 2.0` | **Karena** L1 satu skala punya minimum trivial: spektrum halus (rata-rata kondisional). Multi-scale menahan envelope kasar *dan* transisi cepat, sedangkan **global variance** menghukum pita yang dinamikanya mengempis — persis mekanisme yang membuat suara "buzzy/robotik". (Alternatif MR-STFT butuh bentuk gelombang; vocoder di dalam loop terlalu mahal di CPU, jadi diterapkan di domain mel.) |
| **L_spk** — konsistensi speaker | `1 − cos( probe(mel_pred), z_speaker )` | `λ_spk = 0.0` (OFF) | **Karena** ia butuh vektor timbre referensi `z` dan vektor timbre yang terbaca dari output `probe(mel_pred)`; ia menghukum lebih besar kalau timbre output makin jauh dari referensi. Inilah yang memaksa generator benar-benar **memakai** embedding speaker, bukan menyalin timbre input. |
| **L_probe** — loss probe (saat aktif) | `1 − cos( probe(mel_asli), z_speaker )` | dilatih dengan optimizer sendiri | **Karena** probe harus lebih dulu pandai "membaca timbre dari mel asli"; kalau tidak, L_spk hanya mengejar target yang salah. Probe **hanya** diberi mel asli agar tidak berkolusi dengan generator. |

Catatan operasional: `L_spk` memakai probe yang parameternya **dibekukan** saat langkah generator
(gradien mengalir ke mel, bukan ke probe). Ini meniru skema *classifier/probe loss*: pengukur tidak ikut
bergeser demi menyenangkan pihak yang diukur.

---

## 9. Optimizer & urutan update

| | Generator | Diskriminator | Speaker Probe |
|---|---|---|---|
| Optimizer | AdamW | AdamW | AdamW |
| `lr` | 2e-4 | 2e-4 | 2e-4 |
| `betas` | (0.5, 0.999) | (0.5, 0.999) | (0.5, 0.999) |
| `weight_decay` | 0.0 | 0.0 | 0.0 |
| `grad clip` | 1.0 (norma) | 1.0 | 1.0 |
| Scheduler | **cosine decay** 2e-4 -> 1e-5 per langkah | idem | idem |

Urutan **per langkah (per batch)**:

1. **Langkah D**: hitung `mel_fake = G(content, speaker).detach()`, `L_D` dari `(mel_target, mel_fake)`,
   lalu `zero_grad → backward → step`.
   *Karena* `detach()`: diskriminator dinilai pada mel yang dianggap konstanta, supaya gradiennya tidak
   mengalir balik ke generator. Diskriminator memakai **spectral normalization** supaya tidak mendominasi
   saat generator masih lemah.
2. **Langkah probe** (hanya jika `λ_spk > 0`): `L_probe` dari mel **asli**, optimizer probe.
3. **Langkah G**: hitung ulang `mel_pred = G(...)` (kali ini dengan graf), `L_G` lengkap, lalu
   `zero_grad → backward → step`. Probe dibekukan pada sub-langkah ini.

`lr = 2e-4` + `beta1 = 0.5` adalah resep GAN yang lazim (momentum rendah, karena gradien adversarial
berderau). `weight_decay = 0` karena L2 pada bobot GAN cenderung memperlambat konvergensi tanpa manfaat yang jelas
di skala ini. LR diturunkan **cosine** sepanjang jadwal (bukan konstan) agar fase akhir training lebih
halus, dan gradien di-clip pada norma 1.0 karena sinyal adversarial berderau.

---

## 10. Metrik evaluasi (bukan bagian training)

| Metrik | Definisi | Butuh apa |
|---|---|---|
| **MCD** (dB, kecil lebih baik) | DCT log-mel → ambil `c1..c13`, `MCD = (10/ln10)·√(2·mean_t Σ(Δc)²)` | hanya numpy/scipy |
| **SECS** (besar lebih baik) | cosine antara embedding speaker audio hasil konversi dan embedding audio **enrolment** target; juga dilaporkan terhadap **sumber konten** | encoder speaker XTTS |
| **mel L1** | rata-rata selisih absolut per frame, `L1(mel_pred, mel_target)` | — |

*Karena* MCD membuang `c0` (komponen energi/level), MCD **buta terhadap kesalahan kenyaringan** — itu
sebabnya mel-L1 selalu dilaporkan berpasangan.

Mode evaluasi penting: `--enrollment-mode other-speaker` (konten dari speaker A, embedding dari speaker
B) → SECS menjadi ukuran **transfer timbre** yang sah. Tanpa itu, target dan sumber adalah speaker yang
sama sehingga SECS hanya mengukur "mempertahankan suara", bukan "mengubah suara".

---

## 11. Konfigurasi efektif yang dipakai skrip `.bat`

| Parameter | Nilai | Alasan |
|---|---|---|
| `--data libritts --split train-clean-100` | — | data asli |
| `--content-source hubert` / `--speaker-source xtts` | — | fitur nyata |
| `--segment-seconds` | 4.0 (→ 345 frame mel) | konteks cukup untuk 1 suku kata panjang; 2 s terlalu pendek |
| `--batch-size` | 4 | RAM mesin terbatas (~3,2 GB free); HuBERT hidup di proses yang sama |
| `--num-workers` | 0 | hindari duplikasi bobot HuBERT per worker (RAM) |
| `--min-seconds` | = segment (4 s) | tidak ada *zero-padding* |
| `--val-split dev-clean --val-every 1` | — | kurva MCD pada speaker yang tak dilatih |
| `--max-minutes` | 10 (short) | CPU: ~3,4–3,7 s/langkah |
| `--device` | `auto` → **cpu** | GPU mesin ini (Quadro K620, ~410 MiB kosong) tidak memuat model |

---

## 12. Batasan yang diketahui (jujur, untuk laporan)

1. **Lebar pita 8 kHz** (warisan `mel_fmax=8000` dari vocoder LJSpeech) → suara terasa teredam.
   Solusi: vocoder 24 kHz/44,1 kHz (butuh unduhan + latih ulang domain mel).
2. **Diskriminator satu skala** (belum multi-scale / multi-period).
3. **Belum ada LR decay / AMP / gradient clipping**.
4. **Belum ada resume** — checkpoint sudah menyimpan G, D, probe, dan 3 optimizer, tetapi belum ada kode
   yang memuatnya untuk melanjutkan.
5. **`L_spk` default OFF**; perlu dinyalakan setelah mel-L1 mendatar (dan diukur dengan SECS, bukan dari
   nilai loss-nya).
6. **Ekstraksi HuBERT dilakukan on-the-fly** (belum ada cache fitur) → CPU-bound; cache akan
   mempercepat epoch 2–3×.
7. **Jalur placeholder** (`mel-projection`, `mel-stats`) masih ada untuk uji pipa tanpa bobot — jangan
   dipakai untuk klaim hasil.
