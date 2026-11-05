# First Iteration — State, Temuan, dan Analisis Kegagalan

Dokumen ini mencatat **kondisi apa adanya** pada iterasi pertama: state codebase, state model,
gejala yang terdengar, penyebab yang paling mungkin (dengan bukti/literatur), dan **mengapa pendekatan
sebelumnya tidak berhasil**. Rencana perbaikan ada di `improvement-plan.md`.

Tanggal analisis: setelah run terakhir (150–200 langkah training, CPU).

---

## 1. Ringkasan satu paragraf

Output terdengar "zombie": fonem tidak jelas dan timbre seperti suara rata-rata/mengeram. Ini bukan
satu bug, melainkan **empat penyebab yang bertumpuk**: (a) model masih sangat jauh dari konvergen
(150–200 langkah, sedangkan VC GAN normalnya puluhan ribu langkah), (b) loss reconstruction (L1, λ=45)
jauh mendominasi sehingga generator mengejar **rata-rata kondisional** → spektrum halus dan harmonik
hilang → bunyi *buzzy/robotik*, (c) mel dibatasi **8 kHz** padahal audio sumber 24 kHz, sehingga
konsonan frikatif (/s/, /ʃ/, /f/) kehilangan puncak spektralnya → fonem tidak jelas, dan (d) **tidak ada
tekanan apa pun yang memaksa generator memakai identitas speaker**, sehingga solusi termurah adalah
memakai "suara rata-rata" dari 247 speaker → timbre zombie.

---

## 2. State codebase sekarang

### 2.1 Modul

| Berkas | Isi | Status |
|---|---|---|
| `libritts.py` | `scan_split`, `LibriTTSDataset`, `build_libritts_loader` — audio asli → `(content, speaker, mel)` | jalan |
| `features.py` | `HubertContentExtractor` (768-d, layer 9), `XTTSSpeakerExtractor` (512-d), placeholder `mel-projection`/`mel-stats` | jalan, bobot terpasang |
| `models.py` | `AdaIN1d`, `AdaINResBlock`, `Generator` (5.458.768 param), `PatchDiscriminator` (2.760.001), `SpeakerProbe` (347.008) | jalan |
| `vocoder.py` | `HiFiGANVocoder` — Coqui HiFi-GAN v2, 928.514 param, 22,05 kHz, `mel_fmax=8000` | jalan (beku) |
| `train.py` | loop GAN (hinge/LSGAN), AdamW ×3, validasi, checkpoint, render, `--max-minutes` | jalan |
| `metrics.py` | MCD (c1..c13), SECS (embedding XTTS) | jalan |
| `evaluate.py` | konversi split + MCD/SECS + `report.csv` (timestamp) | jalan |
| `preview.py` | render 1 utterance tetap → `ref.wav` + `gen.wav` untuk A/B yang sah | jalan |
| `arsitektur.md`, `README.md`, `todo.md`, `plan.md` | dokumentasi | sinkron |

### 2.2 Yang dilatih vs yang dibekukan

- **Dilatih:** Generator, Diskriminator, (Speaker Probe hanya jika `--lambda-spk > 0`, default **0**).
- **Beku:** HuBERT (content), XTTS speaker encoder (timbre), vocoder HiFi-GAN.

### 2.3 Konfigurasi efektif (yang dipakai `.bat`)

```
data         : LibriTTS train-clean-100 (33.232 ucapan, 247 speaker), 24 kHz mono
segment      : 4,0 s  → T = 345 frame mel
batch        : 4          num_workers: 0
content      : HuBERT base layer 9 → [B, T, 768], diinterpolasi ke laju frame mel
speaker      : XTTS v2 speaker encoder → [B, 512], L2-normalized
mel          : 22,05 kHz, n_fft 1024, hop 256, 80 bin, fmax 8000, log natural, tanpa normalisasi
loss         : hinge adv (λ=1) + L1 mel (λ=45) + feature matching (λ=2) + L_spk (λ=0, OFF)
optimizer    : AdamW lr 2e-4, betas (0,5;0,999), wd 0, tanpa scheduler
kecepatan    : ~3,4–3,7 detik/langkah (CPU; GPU tidak memungkinkan — Quadro K620, ~410 MiB kosong)
```

---

## 3. State model sekarang (angka terukur)

| Metrik | Nilai | Konteks |
|---|---|---|
| Langkah training total | **150–200** | dua run 10 menit; ditambah beberapa run sanity 2–4 langkah |
| Ucapan yang pernah dilihat | 300–500 (dari 33.232) | 1–1,5 epoch |
| `mel L1` training | 1,379 → 0,905 (run XTTS) · 1,871 → 1,020 (run sebelumnya) | turun, **tetapi turun ≠ membaik** (lihat §5.2) |
| `val MCD` dev-clean | 47,411 → 44,652 dB | 2 epoch |
| `MCD` test-clean | 36,9 dB (ckpt 200 langkah) s/d 49,5 dB (ckpt 150 langkah) | literatur sistem jadi: **5–8 dB** |
| `SECS` target vs sumber | 0,212 vs 0,209 (encoder XTTS) · 0,507 vs 0,558 (encoder resemblyzer) | skala berbeda antar encoder, tak bisa dibandingkan langsung |
| Rentang mel prediksi | min ≈ −12,3 … max ≈ +4,9 | nyaris tidak bergerak dari inisialisasi |
| Rentang mel asli | min −18,4 … max +1,2, **mean −4,5** | referensi |
| Audio yang dihasilkan | 4,01 s, 22,05 kHz mono | terdekode, tidak ada error bentuk tensor |

**Bacaan penting:** tidak ada satu pun tensor error, semua loss *turun*, tetapi MCD masih 5–9× lipat di
atas nilai sistem jadi. Artinya pipa-nya benar secara bentuk, tetapi **jauh dari konvergen dan secara
spektral salah**, bukan "hampir benar".

---

## 4. Gejala yang dilaporkan dan pemetaannya

| Gejala | Terjemahan akustik | Penyebab paling mungkin (§5) |
|---|---|---|
| Fonem tidak jelas ("mbeleber") | puncak frikatif & burst konsonan hilang | 5.3 (band 8 kHz), 5.1 (undertrained) |
| Timbre seperti zombie/mengeram | suara "rata-rata", harmonik lebur | 5.4 (identity averaging), 5.2 (over-smoothing) |
| Terdengar monoton / tanpa intonasi | kontur F0 datar | 5.5 (tidak ada jalur F0) |
| Buzzy/robotik pada bagian bersuara | spektrum halus di-inversi vocoder | 5.2 + 5.7 (feature mismatch vocoder) |

---

## 5. Temuan: mengapa model berbunyi seperti ini

### 5.1 Belum konvergen — selisih 2 orde besaran

VC berbasis GAN yang dilaporkan di literatur dilatih **puluhan ribu sampai ratusan ribu langkah**.
Model ini: **150–200 langkah**, 300–500 ucapan (0,9–1,5% korpus). Pada tahap ini generator praktis
belum keluar dari inisialisasi; MCD 36–49 dB vs 5–8 dB untuk sistem jadi. *Karena* itu, **semua
penyebab lain di bawah ini bekerja pada model yang belum belajar apa pun**, sehingga gejalanya
tampak lebih parah daripada bobot penyebabnya.

> Konsekuensi metodologis: jangan menyimpulkan arsitektur salah dari output 200 langkah. Tapi juga
> jangan menunggu 50 ribu langkah sebelum memperbaiki penyebab struktural (5.2–5.7) — kalau tidak,
> compute-nya habis untuk mencapai plafon yang rendah.

### 5.2 Over-smoothing akibat L1 mendominasi → sumber bunyi "buzzy/robotik"

`λ_mel = 45` membuat L1 mel mendominasi; adversarial hanya λ=1, feature matching λ=2. L1 (atau L2) pada
spektrogram adalah regresi ke **rata-rata kondisional** `E[y|x]`: puncak dan lembah forman diratakan,
kontur F0 ikut rata. Ini adalah mekanisme klasik di literatur VC:

- Kaneko et al. (Interspeech 2017) membangun GAN VC justru untuk **menghindari over-smoothing** dari
  loss level-data ([paper](https://www.isca-archive.org/interspeech_2017/kaneko17_interspeech.pdf)).
- Laporan EPFL menyebut eksplisit: *"Inverting over-smoothed mel-spectrogram gives poor quality and
  buzzy-sounding speech"* ([tesis](https://infoscience.epfl.ch/server/api/core/bitstreams/d5439fb7-3d1f-4ef8-aa55-b495214f7d5f/content)).
- CycleGAN-VC2 menambahkan *two-step adversarial loss* karena L1 rekonstruksi adalah sumber utama
  over-smoothing ([Kaneko et al., ICASSP 2019](http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/publications/Kaneko2019ICASSP05_published.pdf)).
- Catatan penting: `λ_mel = 45, λ_fm = 2` itu resep **vocoder** (HiFi-GAN), bukan resep *converter* —
  saya meminjamnya tanpa menyesuaikan perannya ([MSCGAN](https://dl.acm.org/doi/pdf/10.1145/3625383)).
  Pada vocoder, L1 mel memang besar karena tugasnya inversi; pada converter, bobot segitu membuat G
  malas memproduksi detail.

**Bukti lokal:** mel prediksi tetap berada di rentang sempit (min ≈ −12, max ≈ +5, nyaris tanpa gerakan
dinamis) sementara mel asli punya mean −4,5 dengan variasi jauh lebih lebar. Spektrum "datar" inilah yang
di-inversi vocoder menjadi bunyi buzzy.

### 5.3 Batas pita 8 kHz → fonem tidak jelas

`mel_fmax = 8000` pada audio 24 kHz membuang **seluruh 8–12 kHz** yang sebenarnya ada di data LibriTTS.
Keluhan "fonetik tidak jelas" adalah gejala yang paling langsung dijelaskan oleh ini:

- ICASSP 2024: batas pita menghasilkan kualitas yang disebut *"muffled", "muddy", "mushy"*, dengan
  **hilangnya puncak spektral frikatif** sehingga identifikasi konsonan terganggu
  ([IEEE Xplore](https://ieeexplore.ieee.org/document/10694458)).
- ICASSP 2012: spektrum di sekitar Nyquist "tidak stabil" diwakili mel-cepstrum; energi yang hilang di
  dekat Nyquist membuat suara sintetis cenderung **muffled** dan model malah "belajar" merepresentasikan
  fluktuasi Nyquist ketimbang bagian spektrum penting untuk persepsi
  ([PDF](http://dihana.cps.unizar.es/proceedings/ICASSP/2012/pdfs/0004525.pdf)).
- Literatur bandwidth extension: representasi terkuantisasi **tidak cukup** untuk merekonstruksi /s/ dan
  /z/ yang tajam; efeknya "lisping" ([NAG/DAGA 2009](http://pub.dega-akustik.de/NAG_DAGA_2009/data/articles/000023.pdf)).
  Kontras /s/ vs /ʃ/ hidup di centroid spektral di atas 4–8 kHz — tepat daerah yang kita buang.

*Karena* itu, walau GAN-nya sempurna, **konsonan tetap akan kabur** dengan konfigurasi mel ini.

### 5.4 Rata-rata identitas lintas speaker → timbre "zombie"

Tidak ada satu pun suku loss yang menyebut identitas speaker (L_spk dimatikan; adversarial + L1 + FM
buta terhadap "suara siapa"). Dalam kondisi itu, minimisasi L1 pada target multi-speaker konvergen ke
**suara rata-rata 247 speaker**. Ini pola yang terdokumentasi:

- Diskusi espnet: pelatihan multi-speaker cenderung menghasilkan suara mendekati rata-rata, terutama bila
  informasi speaker hilang di representasi encoder ([issue #5301](https://github.com/espnet/espnet/issues/5301)).
- Ringkasan literatur: "loss generatif standar (MSE pada mel) mendorong model untuk *hedging*, default ke
  profil suara manusia yang halus/'rata-rata' yang memenuhi naturalness tetapi gagal mereplikasi individu"
  ([Speaker Similarity Gap](https://lacuna.tiptreesystems.com/direction/the-speaker-similarity-gap-in-speech-synthesis/txn_2b022953df224ae1ac897866085be710)).
- Model yang berhasil (DeepVoice3, YourTTS) menyuntikkan identitas di **decoder**, bukan di encoder;
  kita sudah melakukannya lewat AdaIN — tetapi AdaIN hanya *mampu*, tidak *terpaksa*, memakai embedding.

**Bukti lokal:** pada evaluasi lintas-speaker, `SECS` ke target (0,21) < `SECS` ke sumber (0,21–0,56 antar
zaman encoder) — output lebih mirip **input** daripada target. Itu tanda conditioning tidak dipakai.

### 5.5 Tidak ada jalur F0/prosodi eksplisit

Statistik instance di AdaIN memang dirancang membuang informasi per-utterance (level, warna), tetapi
**tidak ada jalur pengganti untuk F0**. HuBERT memang menyimpan sebagian prosodi, namun pada layer dalam
(layer 9) informasi pitch makin terprune; dan normalisasi instance di AdaIN membuang statistik
per-utterance. Hasilnya intonasi datar — komponen "zombie" yang lain.

- Lapisan HuBERT: layer 9 adalah titik seimbang fonetik, tetapi layer 6 masih kaya prosodi; kombinasi
  **6+9** dilaporkan memperbaiki kualitas/prosodi tanpa banyak merusak fonetik (GenerTTS,
  [Interspeech 2023](https://www.isca-archive.org/interspeech_2023/cong23_interspeech.pdf)); studi
  HuBERT-Large menemukan layer 22 > layer 24 karena prosodi, layer 24 "lebih mekanis"
  ([arXiv 2409.08039](https://ar5iv.labs.arxiv.org/html/2409.08039)).
- Pemberian F0 eksplisit terbukti menolong: encoder F0 "mengompensasi hilangnya informasi prosodi di
  bottleneck konten" ([VPC 2024](https://www.voiceprivacychallenge.org/vp2024/docs/T33_____KIT-ISL_____Investigating_Voice_Conversion_Architecture_with_Different_Bottleneck_Features_for_the_Voice_Privacy_Challenge_2024.pdf));
  pitch ternormalisasi sebagai input memberi kontrol prosodi eksplisit
  ([TASLP 2024](https://dl.acm.org/doi/pdf/10.1109/TASLP.2024.3439996)); tesis disertasi menyebut
  sebagian besar metode VC **tidak menjamin** disentanglement prosodi tanpa jalur eksitasi/pitch
  ([AAU](https://vbn.aau.dk/files/836023845/PHD_ARB_ONLINE.pdf)).
- AdaIN-VC sendiri lemah pada pitch; varian subband AdaIN-GAN harus menambahkan modul pitch-shift
  ([arXiv 2207.06057](https://ar5iv.labs.arxiv.org/html/2207.06057),
  [AdaIN-VC](https://home.ttic.edu/~jcchou/resource/slides/is19_applier.pdf)).

### 5.6 Interpolasi content 50 Hz → 86 Hz menghaluskan transisi

HuBERT memberi ~50 frame/detik, mel 86 frame/detik; kita **interpolasi linear** agar `T` sama. Secara
durasi benar, tetapi setiap frame content "ditarik" ke 1,72 frame mel sehingga transisi konsonan
(burst 10–30 ms) menjadi landai. Ini memperburuk 5.3: bukan hanya pita tinggi hilang, ketajaman waktu
juga berkurang.

### 5.7 Feature mismatch: vocoder GAN sangat sensitif pada mel "buram"

Vocoder dilatih pada mel **bersih**, tetapi saat inference menerima mel prediksi yang halus. Literatur
menyebut GAN vocoder "sangat sensitif" terhadap distribusi mel yang dihaluskan dan **memperbesar error
kecil menjadi artefak mencolok** (buzzy/metallic)
([feature mismatch](https://lacuna.tiptreesystems.com/direction/mitigating-feature-mismatch-in-cascaded-speech-synthesis/txn_6a5b6c1bb83243e9a428b0c622c49146)).
Artinya buzz yang terdengar bisa berasal dari vocoder **bereaksi** pada over-smoothing 5.2, bukan dari
generator secara langsung.

### 5.8 Diskriminator terlalu lemah untuk menghukum artefak pita tinggi

PatchGAN kita: satu skala, 4 tahap stride-2, `InstanceNorm2d`, hinge, tanpa spectral norm. Tinjauan
MSCGAN menunjukkan *average pooling* pada diskriminator multi-skala **menghapus informasi frekuensi
tinggi**, sehingga generator "tidak diawasi" di pita atas dan detail HF memburuk
([ACM TOG](https://dl.acm.org/doi/pdf/10.1145/3625383)). Dengan D satu skala + tanpa spectral norm,
tidak ada penalti efektif untuk mel landai/harmonik lebur.

### 5.9 Faktor pengali (bukan penyebab utama, tapi memperparah)

- Tanpa LR decay, tanpa gradient clipping → gradien adversarial berderau di lr konstan 2e-4.
- Batch 4 (batas RAM 3,2 GB) → gradien berderau, BatchNorm/InstanceNorm bekerja di batas minimum.
- Ekstraksi HuBERT on-the-fly (+ XTTS speaker encoder 246 ms/ucapan) → ~3,5 s/langkah; 10 menit hanya
  ~170 langkah, jadi tiap eksperimen mahal dan tidak ada iterasi.
- Tidak ada resume → tiap run mulai dari nol.

---

## 6. Mengapa pendekatan sebelumnya tidak berhasil (retrospeksi jujur)

| # | Yang saya lakukan | Mengapa itu tidak menyelesaikan masalah |
|---|---|---|
| 1 | Mengejar *acceptance criteria* "tidak ada `RuntimeError: size mismatch`" | Kriteria itu mengukur **kebenaran bentuk tensor**, bukan kualitas audio. Semua uji bentuk lulus sambil suara tetap rusak. |
| 2 | Memilih komponen berdasarkan metrik yang salah: mengganti encoder speaker karena *margin diskriminasi* naik 0,15 → 0,30 | Margin itu mengukur **kemampuan mengenali** speaker, bukan **kegunaan vektor sebagai kondisi generator**. Target keluhan (audio jelek) tidak tersentuh. |
| 3 | Menambah loss baru (`L_spk`) tanpa instrumen untuk memverifikasi bahwa loss itu mengukur yang dimaksud | Terbukti probe dan generator **saling beradaptasi** (spk 0,93→0,47 dan probe 0,95→0,48 turun bersamaan) — tanda eksploitasi, bukan pembelajaran timbre. Hasilnya audio justru lebih buruk. |
| 4 | Menganggap "loss turun = model membaik" | L1 **selalu** bisa turun dengan cara menghasilkan spektrum makin halus. Jadi penurunan L1 justru konsisten dengan bertambahnya over-smoothing (5.2). |
| 5 | Membandingkan hasil dengan telinga dari file yang **berbeda konten** | `--render-every` menulis crop acak dari utterance acak; dua run tidak pernah membandingkan kata yang sama. Kesimpulan "lebih baik/lebih buruk" jadi tidak valid (baru diperbaiki lewat `preview.py`). |
| 6 | Tidak pernah menetapkan target angka & budget training | Tanpa target (mis. MCD < 8 dB) dan tanpa perkiraan langkah yang dibutuhkan, 150–200 langkah terlihat "sudah training" padahal 2 orde besaran di bawah kebutuhan. |
| 7 | Menerima konfigurasi mel 8 kHz karena "bentuknya cocok" | Konfigurasi itu melekat pada vocoder LJSpeech; dampaknya ke persepsi (frikatif hilang) tidak pernah dievaluasi karena tidak ada uji dengar terinstrumentasi. |
| 8 | Tidak ada baseline pembanding | Tanpa (a) vocoder round-trip mel asli, (b) "suara rata-rata", (c) copy-input, tidak ada cara mengetahui **di mana** plafonnya: apakah di vocoder, di content, atau di conditioning. |
| 9 | Mengesampingkan skala data vs langkah | "Pakai lebih banyak data" menambah ucapan per epoch tetapi **mengurangi** jumlah langkah dalam batas waktu CPU — arah yang berlawanan dengan kebutuhan konvergensi. |

**Pelajaran inti:** iterasi pertama mengoptimasi **kontrak tensor dan angka loss**, bukan **kualitas
persepsi**. Semua perubahan berikutnya harus dimulai dari instrumen yang bisa mengukur kualitas
(spektrogram per pita, F0, uji dengar terkontrol), baru menyentuh model.

---

## 7. Yang *belum* rusak (jangan dibongkar)

- Bentuk tensor & kontrak antar komponen: konsisten, teruji (345 frame, `[B,80,345]`, `[B,512]`, logits
  `[B,1,5,21]`).
- Pipa data: LibriTTS terbaca benar, split speaker bersih (0 tumpang tindih), mel target memakai jalur
  praproses vocoder yang sama.
- Fitur konten (HuBERT layer 9) dan encoder speaker (XTTS) **terbukti bekerja**: konten berbeda antar
  ucapan (cos 0,17), speaker terpisah jelas (cos sama 0,54 vs beda 0,24).
- Reproduksibilitas: seed tetap → dua run 10 menit menghasilkan angka identik.
- Infrastruktur evaluasi (MCD/SECS/preview/validasi) sudah ada dan bisa dipakai mengukur perbaikan.

Kesimpulan: **fondasi pipa sudah benar; yang salah adalah tujuan optimasi, bobot loss, domain spektral,
dan skala training.** Perbaikan ada di `improvement-plan.md`.


---

## 8. Bukti diagnostik terukur (ditambahkan setelah `diagnose.py` dijalankan)

`diagnose.py` sudah dijalankan (utterance `1089_134686_000001_000001`, crop 4 s, checkpoint 150-langkah
pipeline XTTS). Hasilnya **sebagian mengoreksi analisis §5**, dan koreksi itu penting dicatat:

| ukuran | real | roundtrip (mel asli -> vocoder) | meanmel (baseline "suara rata-rata") | model |
|---|---|---|---|---|
| band 0-2 kHz | 0,961 | 0,953 | 0,951 | 0,969 |
| band 2-4 kHz | 0,0328 | 0,0378 | 0,0383 | 0,0221 |
| band 4-6 kHz | 0,00536 | 0,00820 | 0,00496 | 0,00486 |
| band 8-12 kHz | 0,00058 | 0,00063 | 0,00386 | 0,00138 |
| centroid | 1726 Hz | 1763 Hz | 2026 Hz | 1866 Hz |
| F0 std (semitone) | 1,547 | n/a | n/a | **1,018** |
| voiced fraction | 0,322 | 0 | 0 | **0,095** |
| **mel_delta_abs** | **0,4899** | 0,4899 | **0,04201** | **0,4514** |
| mel_bin_std | 1,561 | 1,561 | 0,1328 | 1,782 |

**Tiga kesimpulan yang berubah:**

1. **Vocoder bukan plafon.** `roundtrip` hampir menyamai `real` di semua pita (0,953 vs 0,961 pada 0-2 kHz;
   centroid 1763 vs 1726 Hz). Jadi jalur mel->vocoder mereproduksi isi mel dengan setia. Batas 8 kHz tetap
   batas keras untuk frikatif, tetapi **bukan** penyebab utama output sekarang.
   *(Caveat: klip ini 96% energinya di bawah 2 kHz, jadi ia uji yang lemah untuk pita tinggi.)*
2. **Model TIDAK kolaps ke "suara rata-rata".** `meanmel` memang sangat datar (`mel_delta_abs` 0,042 vs
   real 0,49 = 12x lebih halus), tetapi model punya `mel_delta_abs` 0,451 — **hampir sama dengan real** —
   dan `mel_bin_std` 1,78 (lebih tinggi dari real 1,56). Artinya output model **tidak** over-smooth;
   ia justru berderau/liar.
3. **Yang benar-benar rusak: keperiodikan (voicing).** `voiced fraction` model **0,095** vs real **0,322**
   (3x lebih sedikit bagian bersuara), dan variasi F0 hanya 1,02 semitone vs 1,55. Output sebagian besar
   berupa derau tak bersuara dengan amplop spektral kasar — itulah bunyi "zombie/buzzy" yang terdengar,
   bukan suara halus rata-rata.

**Revisi peringkat penyebab (menggantikan urutan §5):**

| Peringkat baru | Penyebab | Status bukti |
|---|---|---|
| 1 | **Belum konvergen** (150–200 langkah; struktur voicing belum terbentuk) | terukur: voicing 3x terlalu rendah, mel liar |
| 2 | **Belum ada jalur F0 eksplisit** → intonasi datar & voicing lemah | terukur: F0 std 1,02 vs 1,55 |
| 3 | **Over-smoothing berpotensi muncul nanti** (L1 λ=45 mendominasi) | belum terlihat di 150 langkah, tetapi resepnya salah untuk converter |
| 4 | **Batas pita 8 kHz** (plafon konsonan) | roundtrip setia, jadi ini plafon, bukan penyebab sekarang |
| 5 | **Identitas speaker belum dipaksa** | bukti lama (SECS target < sumber) masih berlaku |

Konsekuensi untuk rencana: prioritas bergeser ke **skala training (Fase 2: cache + resume)** lebih dulu,
sementara perubahan loss/F0 (Fase 1) berfungsi mencegah plafon berikutnya. `improvement-plan.md` sudah
diperbarui sesuai itu.


---

## 9. Setelah implementasi iterasi kedua (bukti sebelum/sesudah)

Implementasi yang sudah jalan: `λ_mel` 45 -> 10, **loss detail** (multi-scale + global-variance),
**stream F0**, **spectral norm** pada D, cosine LR 2e-4 -> 1e-5, grad clip 1.0, `--resume`.

Run `train-short.bat` (400 ucapan, batch 4, 4 s, dibatasi 10 menit): **197 langkah** (epoch 1 penuh +
97 langkah epoch 2), 612 s, exit 0. Loss: `mel 1,284 -> 0,929`, `detail 1,468 -> 1,028`
(multi-scale 1,238 -> 0,875; global-var 0,230 -> 0,153), adversarial tetap sehat
(D 1,10 -> 0,42; real 1,15 vs fake -0,96) — jauh lebih seimbang daripada run sebelumnya di mana D
mendominasi (1,73 vs -1,92) lalu loss-nya kolaps.

Diagnostik pada utterance yang sama (`diagnose.py`, checkpoint baru):

| ukuran | real | model lama (λ45, tanpa F0) | **model baru (λ10 + detail + F0)** |
|---|---|---|---|
| voiced fraction | 0,322 | 0,095 (3x terlalu rendah) | **0,425** |
| F0 std (semitone) | 1,547 | 1,018 (datar) | **2,507** |
| mel_delta_abs | 0,4899 | 0,4514 | 0,5161 |
| mel_bin_std | 1,561 | 1,782 | 1,734 |
| centroid | 1726 Hz | 1866 Hz | 1774 Hz |
| band 2-4 kHz | 0,0328 | 0,0221 | 0,0102 |

**Bacaan:** dimensi yang paling merusak persepsi ("hampir tak bersuara / derau") sudah teratasi —
voicing naik 4,5x (0,095 -> 0,425) dan kontur F0 kini bergerak (1,02 -> 2,51 semitone). Perhatikan bahwa
F0 sekarang **melebihi** real (2,51 vs 1,55), jadi mungkin terlalu goyah — kandidat penyetelan berikutnya.
Energi pita 2-4 kHz masih rendah dibanding real (0,010 vs 0,033): konsonan masih kurang tajam, dan ini
konsisten dengan batas 8 kHz (§5.3) yang belum disentuh (butir 1.4).

**Yang belum diuji:** apakah ini terdengar lebih baik (butuh uji dengar `preview_after/`), dan apakah
efeknya bertahan pada skala 10k+ langkah. Jejak pendek tetap tidak membuktikan apa pun soal kualitas
akhir — tetapi metrik objektif yang sebelumnya *salah arah* sekarang bergerak ke arah yang benar.
